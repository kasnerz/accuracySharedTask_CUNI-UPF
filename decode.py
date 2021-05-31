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
from utils.config import Label
from utils import tokenizer

from preprocess import load_games
from generate import Generator

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

tokenizer = tokenizer.Tokenizer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--input_file", default="games.csv", type=str,
        help="Input directory.")
    parser.add_argument("--out_fname", default="out.csv", type=str,
        help="Output file.")
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
    parser.add_argument("--max_length", type=int, default=1024,
        help="Maximum number of tokens per example")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    parser.add_argument("--templates", type=str, default="generated/simple_templates",
        help="Path to generated templates.")
    args = parser.parse_args()

    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
    ecim = ErrorCheckerInferenceModule(args, model_path=model_path)

    g = Generator(args)

    test_games = load_games(args.templates, "test")

    output_path = os.path.join(args.exp_dir, args.experiment, args.out_fname)
    f_out = open(output_path, "w")

    f_out.write(
        '"TEXT_ID","SENTENCE_ID","ANNOTATION_ID","TOKENS","SENT_TOKEN_START","SENT_TOKEN_END","DOC_TOKEN_START","DOC_TOKEN_END","TYPE","CORRECTION","COMMENT"\n'
    )

    with open(args.input_file) as f_in:
        reader = csv.reader(f_in, delimiter=',', quotechar='"')
        # skip header
        header = next(reader)
        error_id = 0

        for row in reader:
            file = row[0]
            test_id = int(row[3])
            text = row[4]

            # misgenerated game
            assert test_id != 515

            if test_id >= 515:
                test_id -= 1

            game_data = test_games[test_id]
            logger.info(f"Processing {file}")
            sentences = sent_tokenize(text)
            doc_token_idx = 0

            for sent_idx, sentence in enumerate(sentences):
                text = g.fetch_text(sentence=sentence, game_data=game_data)
                hyp = sentence
                out = ecim.predict(text=text, hyp=hyp, beam_size=args.beam_size, is_hyp_tokenized=True)

                for token_idx, (token, tag) in enumerate(out):
                    doc_token_idx += 1
                    if tag == Label.O.name:
                        continue

                    logger.info(f"{tag} {token}")
                    error_id += 1
                    out_row = [
                        file + ".txt",
                        sent_idx,
                        error_id,
                        token,
                        token_idx,
                        token_idx,
                        doc_token_idx,
                        doc_token_idx,
                        tag,
                        "",
                        ""
                    ]
                    out_row = [f'"{column}"' for column in out_row if column is not None]
                    f_out.write(",".join(out_row) + "\n")

    f_out.close()