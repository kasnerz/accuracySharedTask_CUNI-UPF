#!/usr/bin/env python3

import os
import argparse
import logging
import json
import random
import re
import torch
import spacy, en_core_web_sm

from utils.json_encoder import CompactJSONEncoder
from utils.config import Label
from utils.tokenizer import Tokenizer as Detokenizer

from nltk import word_tokenize
from num2words import num2words
from text_to_num import alpha2digit

from transformers import AutoTokenizer

from collections import defaultdict
from preprocess import load_games

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True)
        self.detokenizer = Detokenizer()
        self.spacy_nlp = en_core_web_sm.load()

        self.cities = self.load_cities()
        

    def load_cities(self):
        cities = []
        with open("cities.txt") as f:
            for line in f.readlines():
                cities.append(line.rstrip("\n"))
        return cities


    def is_number(self, token):
        if token.isnumeric():
            return True
        elif len(token) > 1 and token.endswith("%") and token[:-1].isnumeric():
            return True

        return False

    def modify_number(self, token):
        num = int(token)
        modified_num = num

        while modified_num == num:
            modified_num = max(0, int(random.gauss(num, 3)))

        return str(modified_num)



    def modify_ordinal(self, token):
        num = alpha2digit(token, "en")

        import pdb; pdb.set_trace()  # breakpoint 7a7ecca3 //


    # def modify_percentage(self, token):
    #     number = re.sub(r"[^\d\.]", '', token)

    #     modified_number = self.modify_number(number)

    #     return modified_number + "%"


    def random_choice_neq(self, l, neq_to):
        assert any([el != neq_to for el in l]), f"Cannot select element not equal to {neq_to} in {l}"

        while True:
            choice = random.choice(l)
            if choice != neq_to:
                return choice


    def modify_entity(self, doc, game_data, entity):
        ent_text = entity[0]
        ent_type = entity[1]
        ent_tokens = entity[4]

        if ent_type == "PERSON":
            random_player = self.random_choice_neq(l=list(game_data.get_players()), neq_to=ent_text)
            tokens = word_tokenize(random_player)
            labels = [Label.NAME for token in tokens]
        elif ent_type == "CARDINAL":
            random_number = self.modify_number(ent_text)
            tokens = [random_number]
            labels = [Label.NUMBER]
        elif ent_type == "GPE":
            random_city = self.random_choice_neq(l=self.cities, neq_to=ent_text)
            tokens = word_tokenize(random_city)
            labels = [Label.NAME for token in tokens]
        elif ent_type == "PERCENT" \
            or ent_type == "TIME" \
            or re.match(r"\d+\s*-\s*\d+", ent_text):      # score, may be recognized as date
            # expressions such as "88% to 77%" or "39 minutes"
            tokens = []
            labels = []
            for token in ent_tokens:
                token = str(token)
                if re.match(r"\d+", token):
                    modified_token = self.modify_number(token)
                    tokens.append(modified_token)
                    labels.append(Label.NUMBER)
                else:
                    tokens.append(token)
                    labels.append(Label.O)
        elif ent_type == "ORDINAL":
            random_number = self.modify_ordinal(ent_text)
            tokens = [random_number]
            labels = [Label.NUMBER]
        elif ent_type == "ORG":
            # misc, usually misrecognized players / teams / cities
            tokens = ent_tokens
            labels = [Label.O for _ in tokens]
        elif ent_type == "DATE":
            logger.warning(f"Unknown entity type {ent_type}: {ent_text}")
            tokens = ent_tokens
            labels = [Label.O for _ in tokens]
            import pdb; pdb.set_trace()  # breakpoint 9cef0122 //

        else:
            logger.warning(f"Unknown entity type {ent_type}: {ent_text}")

            import pdb; pdb.set_trace()  # breakpoint 86655c92 //

            tokens = ent_tokens
            labels = [Label.O for _ in tokens]

        tokens = [str(t) for t in tokens]
        out_txt = " ".join(tokens)

        logger.info(f"[{ent_type}] {ent_text} -> {out_txt}")

        return tokens, labels


    def augment(self, doc, game_data):
        ne = [(ent.text, 
               ent.label_, 
               ent.start, 
               ent.end, 
               [doc[i] for i in range(ent.start, ent.end)]) for ent in doc.ents]
        starts = [x[2] for x in ne]
        ends = [x[3] for x in ne]
        tokens = []
        labels = []
        current_entity = None
        current_entity_started = False

        for i, token in enumerate(doc):
            if i in starts and random.random() < self.args.modification_rate:
                current_entity = ne[starts.index(i)]
                current_entity_started = True
            elif i in ends:
                current_entity = None
            if current_entity is None:
                tokens.append(token)
                labels.append(Label.O)
            elif current_entity_started:
                ent_tokens, ent_labels = self.modify_entity(doc, game_data, current_entity)
                tokens.append(ent_tokens)
                labels.append(ent_labels)


            if current_entity_started:
                current_entity_started = False

        return [(tokens, labels)]

    def generate(self, games, split, samples_per_game):
        data = []

        entity_max_sent_count = 10
        total_sent_count = 15

        for i, game_data in enumerate(games):
            for j in range(samples_per_game):
                all_gt_texts = game_data.get_all()
                entities = list(game_data.entities.keys())
                entity = random.choice(entities)
                entity_gt_texts = game_data.get_entity(entity)

                gt_texts = random.sample(entity_gt_texts, min(entity_max_sent_count, len(entity_gt_texts)))
                gt_texts += random.sample(all_gt_texts, total_sent_count - len(gt_texts))

                assert len(gt_texts) == total_sent_count

                sents = random.sample(gt_texts, random.randint(1,4))
                sent = " ".join(sents)

                gt_tokens = word_tokenize(" ".join(gt_texts))
                sep_token = [self.tokenizer.sep_token]

                gt_labels = [Label.O for _ in range(len(gt_tokens)+1)] # also for the sep_token

                doc = self.spacy_nlp(sent)
                augmented_sents = self.augment(doc, game_data)

                for aug_tokens, aug_labels in augmented_sents:
                    # task data do not contain "%", only "percent" 
                    aug_tokens = [str(t).replace("%", "percent") for t in aug_tokens]

                    example = {
                        "text" : gt_tokens + sep_token + aug_tokens,
                        "labels" : gt_labels + aug_labels
                    }

                    data.append(example)

            if i > 0 and i % 10000 == 0:
                logger.info(f"{i} examples generated.")

        with open(os.path.join(self.args.data_dir, self.args.output, f"{split}.json"), "w") as f:
            s = json.dumps({"data" : data}, ensure_ascii=False, indent=4, cls=CompactJSONEncoder)
            f.write(s)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=str, default="generated/simple_templates",
        help="Path to generated templates.")
    parser.add_argument("--data_dir", type=str, default="data",
        help="Path to data.")
    parser.add_argument("--output", type=str, required=True,
        help="Name of the output directory.")
    parser.add_argument("--samples_per_game", type=int, default=10,
        help="Training samples generated per game.")
    parser.add_argument("--modification_rate", type=float, default=0.5,
        help="Proportion of modified entities.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.join(args.data_dir, args.output), exist_ok=True)
    g = Generator(args)

    for split in ["train", "dev", "test"]:
        games = load_games(args.templates, split)
        g.generate(games, split, args.samples_per_game)
