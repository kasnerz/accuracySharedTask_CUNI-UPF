#!/usr/bin/env python3

import os
import argparse
import logging
import json
import random
import re

from collections import defaultdict


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class GameData:
    def __init__(self):
        self.entities = defaultdict(list)
        self.teams = set()
        self.players = set()
        self.texts = []

    def get_all(self):
        return [x[0] for x in self.texts]

    def get_entity(self, entity):
        return self.entities[entity]

    def get_teams(self):
        return self.teams

    def get_players(self):
        return self.players

    def get_category(self, category):
        return [x[0] for x in self.texts if x[1] == category]

    def add_text(self, text, category):
        entities = identify_entities(text)

        for entity in entities:
            self.entities[entity].append(text)

            if category in ["game", "team"]:
                self.teams.add(entity)
            elif category == "player":
                self.players.add(entity)

        self.texts.append((text, category))


def identify_entities(sent):
    entity = []

    for token in sent.split():
        if token[0].isupper():
            entity.append(token)
        else:
            break

    return [" ".join(entity)]



def load_games(template_path, split="test"):
    games = []
    game_data = None
    category = None
    entity = None

    skipped_ids = []
    game_id = 0

    with open(template_path + f"/log_{split}.txt") as f:
        for line in f.readlines():
            line = line.rstrip("\n")
            skipped_id = re.search(r"Input #(\d+) not processed", line)

            if line.startswith("=== Game"):
                if game_data:
                    games.append((int(game_id), game_data))
                    game_id += 1

                    while game_id in skipped_ids:
                        logger.info(f"Skipping id {game_id}")
                        game_id += 1
                game_data = GameData()
            elif "Game Data" in line:
                category = "game"
            elif "Player Data" in line:
                category = "player"
            elif "Team Data" in line:
                category = "team"
            elif skipped_id is not None:
                skipped_ids.append(int(skipped_id.group(1)))
            elif line == "" or category is None:
                continue
            else:
                game_data.add_text(line, category)

    games.append((int(game_id),game_data))
    return games


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=str, default="generated/simple_templates",
        help="Path to generated templates.")
    args = parser.parse_args()

    games = load_games(args.templates)

