#!/usr/bin/env python3

import os
import argparse
import logging
import json
import random
import re

from collections import defaultdict
from utils.rotowire import Rotowire

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class GameData:
    """
    Data structure for generated sentences about a game
    """
    def __init__(self, game_id, rotowire):
        self.teams = rotowire.get_teams(game_id)
        self.players = rotowire.get_players(game_id)
        self.texts = []

    def get_all(self):
        return self.texts

    def get_teams(self):
        return self.teams

    def get_players(self):
        return self.players

    def add_text(self, text):
        self.texts.append(text)


def load_games(template_path, rotowire_dir, split="test", normalize=True):
    """
    Load the generated sentences from the text file into data structures
    """
    games = []
    game_data = None
    skipped_ids = []
    game_id = 0
    rotowire = Rotowire(rotowire_dir, split)

    with open(template_path + f"/{split}.txt") as f:
        for line in f.readlines():
            line = line.rstrip("\n")

            if normalize:
                # spaces around a dash
                line = re.sub(r"(\d+)-(\d+)", r"\1 - \2", line)
                # spaces around a slash
                line = re.sub(r"(\d+)/(\d+)", r"\1 / \2", line)
                # % -> percent
                line = re.sub(r"(\d+)%", r"\1 percent", line)
                # only single spaces
                line = re.sub(r"(\s)\s+", r"\1", line)

            # for some games, an output could not be generated -> skip
            skipped_id = re.search(r"Input #(\d+) not processed", line)

            if line.startswith("=== Game"):
                if game_data:
                    games.append(game_data)
                    game_id += 1

                    while game_id in skipped_ids:
                        logger.info(f"Game id {game_id} skipped")
                        games.append(None)
                        game_id += 1
                game_data = GameData(game_id, rotowire)
            elif skipped_id is not None:
                skipped_ids.append(int(skipped_id.group(1)))
            elif line == "" or not game_data or re.search(r"(Game|Player|Team) Data", line):
                continue
            else:
                game_data.add_text(line)

    games.append(game_data)

    return games


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=str, required=True,
        help="Path to generated templates.")
    parser.add_argument("--rotowire_dir", type=str, default="rotowire",
        help="Path to the original Rotowire dataset.")
    args = parser.parse_args()

    games = load_games(args.templates, rotowire_dir=rotowire_dir)

