#!/usr/bin/env python3

import os
import argparse
import logging
import json
import random
import re

from utils.json_encoder import CompactJSONEncoder
from utils.config import Label

from nltk import word_tokenize

from transformers import AutoTokenizer
import torch

from collections import defaultdict
from preprocess import load_games

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True)

    def is_number(self, token):
        if token.isnumeric():
            return True
        elif len(token) > 1 and token.endswith("%") and token[:-1].isnumeric():
            return True

        return False

    def modify_number(self, token):
        suffix = ""
        if token.endswith("%"):
            token = token[:-1]
            suffix = "%"

        num = int(token)
        modified_num = num

        while modified_num == num:
            modified_num = max(0, int(random.gauss(num, 3)))

        return str(modified_num) + suffix

    def generate(self, games, split):
        data = []

        for i, game in enumerate(games):
            player = random.choice(list(game["player"].keys()))
            gt = random.sample(game["player"][player], min(len(game["player"][player]), 15))
            sent = random.sample(gt, min(len(gt), 3))


            import pdb; pdb.set_trace()  # breakpoint 2df55309 //

            gt_tokens = " ".join(gt).split()
            sep_token = [self.tokenizer.sep_token]
            sent_tokens = " ".join(sent).split()
            modified_sent_tokens = []

            gt_labels = [Label.O for _ in range(len(gt_tokens)+1)] # also for the sep_token
            modified_sent_labels = []
            modification_rate = 0.5

            for token in sent_tokens:
                label = Label.O
                if self.is_number(token):
                    if random.random() < modification_rate:
                        orig_token = token
                        token = self.modify_number(token)
                        logger.info(f"{orig_token} -> {token}")
                        label = Label.NUMBER

                modified_sent_labels.append(label)
                modified_sent_tokens.append(token)

            example = {
                "text" : gt_tokens + sep_token + modified_sent_tokens,
                "labels" : gt_labels + modified_sent_labels
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
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.join(args.data_dir, args.output), exist_ok=True)
    g = Generator(args)

    for split in ["train", "dev", "test"]:
        games = load_games(args.templates, split)
        g.generate(games, split)
