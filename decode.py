#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import re
import torch
import json
import csv
import pytorch_lightning as pl

from pprint import pprint as pp
from model import ErrorCheckerInferenceModule
from nltk import sent_tokenize
from utils.utils import Label
from utils import tokenizer
from utils.sentence_scorer import SentenceScorer

from preprocess import load_games
from generate import Generator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

tokenizer = tokenizer.Tokenizer()


class Decoder:
    def __init__(self, args):
        self.args = args
        ec_model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)

        self.ec = ErrorCheckerInferenceModule(args, model_path=ec_model_path)
        self.ss = SentenceScorer()


    def process_input_file(self, row, test_games, f_out):
        """
        Produces annotations for a single game.
        """
        file = row[0]
        test_idx = int(row[3])
        text = row[4]

        game_data = test_games[test_idx]

        logger.info(f"Processing {file}")
        sentences = sent_tokenize(text)
        doc_token_idx = 0

        for sent_idx, sentence in enumerate(sentences):
            text = self.ss.retrieve_ctx(sentence=sentence, game_data=game_data, cnt=self.args.ctx)
            hyp = sentence
            out = self.ec.predict(text=text, hyp=hyp, beam_size=self.args.beam_size, is_hyp_tokenized=True)

            for token_idx, (token, tag) in enumerate(out):
                doc_token_idx += 1
                self.error_id += 1

                # skip if OK
                if tag == Label.O.name:
                    continue

                logger.info(f"{tag} {token}")
                
                # write an error annotation (need to be careful about indices starting from 1)
                out_row = [
                    file + ".txt",
                    sent_idx+1,
                    self.error_id,
                    token,
                    token_idx+1,
                    token_idx+1,
                    doc_token_idx,
                    doc_token_idx,
                    tag,
                    "",
                    ""
                ]
                out_row = [f'"{column}"' for column in out_row if column is not None]
                f_out.write(",".join(out_row) + "\n")


    def decode(self):
        """
        Produces error annotations for the games.csv file
        """
        self.error_id = 0
        templates_path = f"./context/{args.templates}"

        test_games = load_games(templates_path, rotowire_dir=self.args.rotowire_dir, split="test")
        output_path = os.path.join(args.exp_dir, args.experiment, args.out_fname)
        
        with open(args.input_file) as f_in, open(output_path, "w") as f_out:
            reader = csv.reader(f_in, delimiter=',', quotechar='"')
            # skip header of games.csv
            next(reader)

            # write a submission header
            f_out.write(
                '"TEXT_ID","SENTENCE_ID","ANNOTATION_ID","TOKENS","SENT_TOKEN_START",'
                '"SENT_TOKEN_END","DOC_TOKEN_START","DOC_TOKEN_END","TYPE","CORRECTION","COMMENT"\n'
            )
            for row in reader:
                # each game is in a separate file
                self.process_input_file(row, test_games, f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--input_file", default="games.csv", type=str,
        help="Input directory.")
    parser.add_argument("--out_fname", default="out.csv", type=str,
        help="Output file.")
    parser.add_argument("--ctx", type=int, default=None, required=True,
        help="Number of sentences retrieved for the context.")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name.")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=4, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=1, type=int,
        help="Beam size.")
    parser.add_argument("--gpus", default=1, type=int,
        help="Number of GPUs.")
    parser.add_argument("--max_length", type=int, default=512,
        help="Maximum number of tokens per example")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    parser.add_argument("--templates", type=str, default=None,
        help="Type of templates (simple / compact).")
    parser.add_argument("--rotowire_dir", type=str, default="rotowire",
        help="Path to the original Rotowire dataset.")
    args = parser.parse_args()

    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    d = Decoder(args)
    d.decode()