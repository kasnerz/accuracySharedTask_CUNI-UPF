#!/usr/bin/env python3

import numpy as np
import os
import logging
import re
import json
import argparse
import pytorch_lightning as pl
import torch

import torch.nn as nn
import torch.nn.functional as F
import random
import sacrebleu

from nltk import word_tokenize
from torch.utils.data import DataLoader, Dataset

from collections import defaultdict
from datasets import load_dataset, dataset_dict, Dataset

from torch.nn.utils.rnn import pad_sequence
from utils.config import Label
from collections import OrderedDict

from transformers import (
    AdamW,
    Adafactor,
    AutoModelForTokenClassification,
    AutoConfig,
    AutoTokenizer,
    get_scheduler
)

logger = logging.getLogger(__name__)


class ErrorCheckerDataModule(pl.LightningDataModule):
    def __init__(self, args, model_name=None):
        super().__init__()
        self.args = args
        self.model_name = model_name or self.args.model_name

        # disable the "huggingface/tokenizers: The current process just got forked" warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       use_fast=True,
                                                       add_prefix_space=True)

    def setup(self, stage):
        data_dir = os.path.join("data", self.args.dataset)

        if stage == "fit":
            splits = ["train", "dev"]
        elif stage == "predict":
            splits = ["dev", "test"]

        self.dataset = {
            split : load_dataset("json", data_files=os.path.join(data_dir, f"{split}.json"), field="data", split="train") 
                    for split in splits
        }

        for split in self.dataset.keys():
            for column in self.dataset[split].column_names:
                if column not in ["text", "labels"]:
                    self.dataset[split] = self.dataset[split].remove_columns(column)

            self.dataset[split] = self.dataset[split].map(
                self._convert_to_features,
                batched=True,
                remove_columns=['labels'],
            )
            self.dataset[split].set_format(
                type="torch",
                columns=[
                    "attention_mask", "input_ids", "labels"
                ])


    def _convert_to_features(self, example_batch, indices=None):
        text = example_batch["text"]

        features = self.tokenizer(text, 
            is_split_into_words=True, 
            return_offsets_mapping=True,
            max_length=self.args.max_length,
            truncation=True)

        features['labels'] = self._align_labels_with_tokens(
            features, example_batch['labels'])

        return features


    def _align_labels_with_tokens(self, features, labels):
        aligned_labels_batch = []
        label_map = Label.label2id()
        do_not_care_label = -100

        for b in range(len(labels)):
            aligned_labels = []

            # compute loss only for the hypothesis, skip the context
            active = False
            
            for i, (input_id, word) in enumerate(zip(features["input_ids"][b], features.words(b))):
                aligned_label = do_not_care_label

                if word is not None:
                    label = labels[b][word]

                # align with word in case a new word started
                # word is None for BOS and EOS
                if active and word is not None and features.words(b)[i-1] != word:
                    aligned_label = label_map[labels[b][word]]

                if input_id == self.tokenizer.sep_token_id:
                    active = True

                aligned_labels.append(aligned_label)

            assert len(features['input_ids'][b]) == len(aligned_labels)
            aligned_labels_batch.append(aligned_labels)

        return aligned_labels_batch


    def train_dataloader(self):
        return DataLoader(self.dataset['train'],
                          batch_size=self.args.batch_size,
                          num_workers=self.args.max_threads,
                          collate_fn=self._pad_sequence,
                          )

    def val_dataloader(self):
        return DataLoader(self.dataset['dev'],
                          batch_size=self.args.batch_size,
                          num_workers=self.args.max_threads,
                          collate_fn=self._pad_sequence)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          batch_size=self.args.batch_size,
                          num_workers=self.args.max_threads,
                          collate_fn=self._pad_sequence)


    def _pad_sequence(self, batch):
        batch_collated = {}

        paddings = {
            "input_ids" : self.tokenizer.pad_token_id,
            "attention_mask" : 0,
            "labels" : -100
        }

        for key in ["input_ids", "attention_mask", "labels"]:
            elems = [x[key] for x in batch]
            elems_pad = pad_sequence(elems, batch_first=True, padding_value=paddings[key])
            batch_collated[key] = elems_pad

        return batch_collated



class ErrorChecker(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(
                self.args.model_name,
                num_labels=len(Label),
                id2label=Label.id2label(),
                label2id=Label.label2id()
            )

        self.model = AutoModelForTokenClassification.from_pretrained(args.model_name,
                                              config=config)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                                       use_fast=True,
                                                       add_prefix_space=True)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs["loss"]

        self.log('loss/train', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       labels=labels)
        loss = outputs["loss"]

        self.log('loss/val', loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon,
                          betas=(self.args.adam_beta1, self.args.adam_beta2))

        total_steps = self.args.max_steps if self.args.max_steps else len(
            self.train_dataloader()) * self.args.max_epochs
        warmup_steps = total_steps * self.args.warmup_proportion

        scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)

        logger.info(f"Using Adam optimizer")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument("--learning_rate", default=5e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.997, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--weight_decay", default=0.00, type=float)

        return parser



class ErrorCheckerInferenceModule:
    def __init__(self, args, model_path):
        self.args = args
        self.model = ErrorChecker.load_from_checkpoint(model_path)
        self.model.freeze()
        logger.info(f"Loaded model from {model_path}")

        self.model_name = self.model.model.name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       use_fast=True,
                                                       add_prefix_space=True)
        if hasattr(self.args, "gpus") and self.args.gpus > 0:
            self.model.cuda()
        else:
            logger.warning("Not using GPU")

    def beam_search_decoder(self, logits, k):
        batch_size, seq_length, vocab_size = logits.shape
        log_prob, indices = logits[:, 0, :].topk(k, sorted=True)
        indices = indices.unsqueeze(-1)


        for i in range(1, seq_length):
            log_prob = log_prob.unsqueeze(-1) + logits[:, i, :].unsqueeze(
                1).repeat(1, k, 1)
            log_prob, index = log_prob.view(batch_size, -1).topk(k,
                                                                 sorted=True)
            indices = torch.cat(
                [indices, index.unsqueeze(-1) % vocab_size], dim=-1)

        return indices

    def predict(self, text, hyp, beam_size=1, is_hyp_tokenized=False):
        text_tokens = word_tokenize(text)

        if is_hyp_tokenized:
            hyp_tokens = hyp.split()
        else:
            hyp_tokens = word_tokenize(hyp)

        tokens = text_tokens + [self.tokenizer.sep_token] + hyp_tokens

        inputs = self.tokenizer(tokens, 
                        return_tensors='pt',
                        return_offsets_mapping=True,
                        max_length=self.args.max_length,
                        truncation=True,
                        is_split_into_words=True)

        if hasattr(self.args, "gpus") and self.args.gpus > 0:
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()

        logits = self.model.model(input_ids=inputs["input_ids"]).logits
        
        if beam_size > 1:
            predictions = self.beam_search_decoder(logits, beam_size)
            predictions = predictions[0][0].cpu().numpy()
        else:    
            predictions = np.argmax(logits.cpu().numpy(), axis=2)[0]

        id2label = Label.id2label()

        offset_mapping = inputs["offset_mapping"][0]
        word_indices = [idx for idx, offset in enumerate(offset_mapping) if offset[0]==1]
        hyp_word_indices = word_indices[-len(hyp_tokens):]
        hyp_labels = predictions[hyp_word_indices]
        hyp_tagged_tokens = [(token, id2label[label]) for token, label in zip(hyp_tokens, hyp_labels)]

        return hyp_tagged_tokens


