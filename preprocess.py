#!/usr/bin/env python3

import os
import argparse
import logging
import json
import random
import re

from collections import defaultdict


def identify_entity(sent):
    entity = []

    for token in sent.split():
        if token[0].isupper():
            entity.append(token)
        else:
            break

    return " ".join(entity)



def load_games(template_path, split="test"):
    games = []
    game = None
    category = None
    entity = None

    with open(template_path + f"/log_{split}.txt") as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            if line.startswith("=== Game"):
                if game:
                    games.append(game)
                game = defaultdict(lambda: defaultdict(list))
            elif "Game Data" in line:
                category = "game"
            elif "Player Data" in line:
                category = "player"
            elif "Team Data" in line:
                category = "team"
            elif line == "" or category is None:
                continue
            else:
                entity = identify_entity(line)
                game[category][entity].append(line)

    games.append(game)
    return games

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=str, default="generated/simple_templates",
        help="Path to generated templates.")
    args = parser.parse_args()

    games = load_games(args.templates)

