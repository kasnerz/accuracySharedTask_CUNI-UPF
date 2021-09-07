#!/usr/bin/env python
"""
Command-line interactive mode for running the model.
"""

import argparse
import logging
import numpy as np
import os
import re
import torch

import pytorch_lightning as pl

from pprint import pprint as pp
from model import ErrorCheckerInferenceModule
from preprocess import load_games
from utils.sentence_scorer import SentenceScorer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name.")
    parser.add_argument("--rotowire_dir", type=str, default="rotowire",
        help="Path to the original Rotowire dataset.")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=1, type=int,
        help="Beam size.")
    parser.add_argument("--gpus", default=1, type=int,
        help="Number of GPUs (NOTE: only 0 or 1 gpus supported for now).")
    parser.add_argument("--ctx", type=int, required=True,
        help="Number of sentences retrieved for the context.")
    parser.add_argument("--max_length", type=int, default=512,
        help="Maximum number of tokens per example")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    parser.add_argument("--game_idx", default=None, type=int,
        help="Retrieve the references automatically from game_id.")
    parser.add_argument("--templates", type=str, default=None,
        help="Type of templates (simple / compact).")
    parser.add_argument("--is_tokenized", action="store_true",
        help="Input is tokenized, split on spaces.")

    args = parser.parse_args()
    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
    ecim = ErrorCheckerInferenceModule(args, model_path=model_path)

    templates_path = f"./context/{args.templates}"
    game_idx = args.game_idx

    if game_idx is None:
        # custom context
        text = input("[Context]: ")
    else:
        # retrieve context automatically based on the game id
        ss = SentenceScorer()
        test_games = load_games(templates_path, rotowire_dir=args.rotowire_dir, split="test")
        game_data = test_games[game_idx]

    while True:
        hyp = input("[Sentence]: ")

        if game_idx is not None:
            # print the automatically retrieved context
            text = ss.retrieve_ctx(hyp, game_data, cnt=args.ctx)
            print("[Context]:", text)

        out = ecim.predict(text=text, hyp=hyp, beam_size=args.beam_size, is_hyp_tokenized=args.is_tokenized)
        print("[Out]:")
        pp(out)
        print("============")


