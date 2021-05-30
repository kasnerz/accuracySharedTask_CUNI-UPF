#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import re
import torch

import pytorch_lightning as pl

from pprint import pprint as pp
from model import ErrorCheckerInferenceModule


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--experiment", type=str, required=True,
        help="Experiment name.")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=5, type=int,
        help="Beam size.")
    parser.add_argument("--gpus", default=1, type=int,
        help="Number of GPUs.")
    parser.add_argument("--max_length", type=int, default=1024,
        help="Maximum number of tokens per example")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    args = parser.parse_args()

    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)

    model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
    ecim = ErrorCheckerInferenceModule(args, model_path=model_path)

    text = input("[Text]: ")

    while True:
        hyp = input("[Hyp]: ")

        out = ecim.predict(text=text, hyp=hyp, beam_size=args.beam_size)
        print("[Out]:")
        pp(out)
        print("============")


