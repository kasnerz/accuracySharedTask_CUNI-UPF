#!/usr/bin/env python3

import os
import argparse
import logging
import json
import random
import re
import torch
import spacy, en_core_web_sm
import csv

from utils.json_encoder import CompactJSONEncoder
from utils.config import Label
from utils.tokenizer import Tokenizer as Detokenizer
from utils.sentence_scorer import SentenceScorer

from nltk import word_tokenize, sent_tokenize
from num2words import num2words
from text_to_num import alpha2digit

from transformers import AutoTokenizer

from collections import defaultdict
from preprocess import load_games

from pprint import pprint as pp

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class Generator:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True, add_prefix_space=True)
        self.detokenizer = Detokenizer()
        self.spacy_nlp = en_core_web_sm.load()
        self.cities = self.load_cities()
        self.days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        self.rotowire = defaultdict(None)
        self.ss = SentenceScorer()
        for split in ["test", "dev", "train"]:
            with open(os.path.join(self.args.rotowire_dir, f"{split}.json")) as f:
                self.rotowire[split] = json.load(f)


    def load_cities(self):
        """
        Load the list of team cities extracted from the Rotowire dataset
        """
        cities = []
        with open("cities.txt") as f:
            for line in f.readlines():
                cities.append(line.rstrip("\n"))
        return cities


    def modify_number(self, text):
        """
        Modify a cardinal number (integer). Gaussian noise with a cut-off at
        zero is applied until the number is different from the original number.

        If `text` is not digit-only, alpha2digit is used to transform the numbers 
        in text to digit-only variants and transformed back to text after modifications.
        """
        text = str(text)

        if not all([l.isnumeric() for l in text]):
            to_lex = True
            text = alpha2digit(text, "en")
        else:
            to_lex = False

        # TODO change to findall(), apply the modifications separately
        num = re.sub(r"[^\d]", "", str(text))
        
        if not num:
            return text

        num = int(num)
        modified_num = num

        # apply Gaussian noise with sigma=3 until the number differs
        # negative numbers do not occur in texts -> threshold at zero
        while modified_num == num:
            modified_num = max(0, int(random.gauss(num, 3)))

        if to_lex:
            modified_num = num2words(int(modified_num), to='cardinal')

        return str(modified_num)


    def modify_ordinal(self, token):
        # TODO rewrite
        if not token[0].isnumeric():
            to_lex = True
            num = alpha2digit(token, "en")

            # alpha2digit does not handle these
            extra_ordinals = {
                "first" : "1",
                "second" : "2",
                "third" : "3"
            }
            if num in extra_ordinals.keys():
                num = extra_ordinals[num]
        else:
            to_lex = False
            num = token

        num = re.sub(r"[^\d]", "", num)

        # conversion to digits not successful
        if not num:
            return token

        num = int(num)
        modified_num = self.modify_number(num)

        if to_lex:
            modified_num = num2words(int(modified_num), to='ordinal')

        return modified_num



    def random_choice_neq(self, l, neq_to):
        """
        Return a random element from a list. The element must be different from the given element `neq_to`.
        """
        assert any([el != neq_to for el in l]), f"Cannot select element not equal to {neq_to} in {l}"

        while True:
            choice = random.choice(l)
            if choice != neq_to:
                return choice


    def modify_entity(self, doc, game_data, entity):
        """
        Change an entity to an errorneous one.

        doc - output of Spacy NER NLP pipeline
        game_data - GameData object for the relevant game
        entity - (text, label, start, end, list_of_tokens) as identified Spacy NER
        """
        ent_text = entity[0]
        ent_type = entity[1]
        ent_tokens = entity[4]

        if ent_text in self.days_of_week:
            # change a day of week to a random day
            random_day = self.random_choice_neq(l=self.days_of_week, neq_to=ent_text)
            tokens = [random_day]
            labels = [Label.NAME]
        elif ent_type == "PERSON":
            # swap a player with another player present in the game
            random_player = self.random_choice_neq(l=list(game_data.get_players()), neq_to=ent_text)
            tokens = word_tokenize(random_player)
            labels = [Label.NAME for token in tokens]
        elif ent_type == "CARDINAL":
            # try to modify a cardinal number (assign OK if modificiation not succesful)
            modified_number = self.modify_number(ent_text)

            if modified_number != ent_text:
                tokens = [modified_number]
                labels = [Label.NUMBER]
            else:
                tokens = [ent_text]
                labels = [Label.O]
        elif ent_type == "GPE":
            # GPE - usually a name of the city; select another city from all the cities
            # in the Rotowire dataset
            random_city = self.random_choice_neq(l=self.cities, neq_to=ent_text)
            tokens = word_tokenize(random_city)
            labels = [Label.NAME for token in tokens]
        elif ent_type == "PERCENT" \
            or ent_type == "TIME" \
            or ent_type == "QUANTITY" \
            or re.match(r"\d+\s*-\s*\d+", ent_text): # score, may be recognized as date

            # modify cardinal numbers token-wise 
            # TODO merge with CARDINAL, change modify_cardinal to take into account surrounding text
            tokens = []
            labels = []
            # can be a multi-word expression such as "88% to 77%" or "39 minutes"
            for token in ent_tokens:
                token = str(token)
                if re.match(r"\d+", token):
                    modified_token = self.modify_number(token)
                    if modified_token != token:
                        tokens.append(modified_token)
                        labels.append(Label.NUMBER)
                    else:
                        tokens.append(token)
                        labels.append(Label.O)
                else:
                    tokens.append(token)
                    labels.append(Label.O)
        elif ent_type == "ORDINAL" \
            or "half" in ent_text \
            or "quarter" in ent_text:               # "X-th quarter/half", may be recognized as date
            tokens = []
            labels = []
            for token in ent_tokens:
                token = str(token)
                modified_token = self.modify_ordinal(token)

                if modified_token != token:
                    tokens.append(modified_token)
                    labels.append(Label.NUMBER)
                else:
                    tokens.append(token)
                    labels.append(Label.O)
        elif ent_type == "ORG" or ent_type == "FAC":
            random_team = self.random_choice_neq(l=list(game_data.get_teams()), neq_to=ent_text)
            tokens = word_tokenize(random_team)
            labels = [Label.NAME for token in tokens]
        elif ent_type in ["NORP", "PRODUCT", "EVENT", "DATE", "LOC"]:
            # logger.warning(f"Skipping {ent_type}: {ent_text}")
            tokens = ent_tokens
            labels = [Label.O for _ in tokens]
        else:
            logger.warning(f"Unknown entity type {ent_type}: {ent_text}")
            tokens = ent_tokens
            labels = [Label.O for _ in tokens]

        tokens = [str(t) for t in tokens]
        labels = [l.name for l in labels]
        out_txt = " ".join(tokens)

        logger.info(f"[{ent_type}] {ent_text} -> {out_txt} {labels}")

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
                # modify only a portion of named entities defined by the modification_rate argument
                current_entity = ne[starts.index(i)]
                current_entity_started = True
            elif i in ends:
                current_entity = None
            if current_entity is None:
                tokens.append(token)
                labels.append(Label.O.name)
            elif current_entity_started:
                # modified entity may have a different number of tokens
                # -> new tokens are added, remaining original tokens are skipped
                ent_tokens, ent_labels = self.modify_entity(doc, game_data, current_entity)
                tokens += ent_tokens
                labels += ent_labels
                current_entity_started = False

        # only a single augmented sentence per input sentence
        # TODO remove for loop?
        return [(tokens, labels)]

    def generate(self, games, split):
        data = []

        start = self.args.start or 0
        end = self.args.end or len(games)

        for game_id in range(start, end):
            game_data = games[game_id]
            rotowire_game = self.rotowire[split][game_id]
            rotowire_summary = rotowire_game["summary"]
            rotowire_summary_sents = sent_tokenize(" ".join(rotowire_summary))

            for sent in rotowire_summary_sents:
                sent_detok = self.detokenizer.detokenize(sent)
                gt_texts = self.ss.retrieve_ctx(sent_detok, game_data)

                import pdb; pdb.set_trace()  # breakpoint f09e9b89 //

                gt_tokens = word_tokenize(gt_texts)

                sep_token = [self.tokenizer.sep_token]
                gt_labels = [Label.O.name for _ in range(len(gt_tokens)+1)] # also for the sep_token

                # run the Spacy NLP pipeline for identifying named entities
                doc = self.spacy_nlp(sent_detok)
                augmented_sents = self.augment(doc, game_data)

                for aug_tokens, aug_labels in augmented_sents:
                    tokens_all = gt_tokens + sep_token + aug_tokens
                    # there is not "%" sign in the task data
                    # the backslash may break the resulting JSON
                    tokens_all = [str(token).replace("%", "percent").replace("\\","") for token in tokens_all]
                    labels_all = gt_labels + aug_labels

                    assert len(tokens_all) == len(labels_all)

                    example = {
                        "text" : tokens_all,
                        "labels": labels_all
                    }
                    data.append(example)

                    if len(data) > 0 and len(data) % 100 == 0:
                        logger.info(f"{len(data)} examples generated.")

        out_fname = f"{split}.json"

        if self.args.start or self.args.end:
            # partial file for parallelization, will be merged
            out_fname = f"{split}-{self.args.start:04d}-{self.args.end:04d}.json"

        with open(os.path.join(self.args.data_dir, self.args.output, out_fname), "w") as f:
            s = json.dumps({"data" : data}, ensure_ascii=False, indent=4, cls=CompactJSONEncoder)
            f.write(s)


    def extract_annotated(self, games):
        """
        Generate training data from the annotated data for the task.
        """
        splits = ["train", "dev", "test"]
        data = {s : [] for s in splits}

        # the games from Rotowire are used for retrieving the context for the annotated errors
        with open(self.args.annotations) as f_anno, open(self.args.games) as f_games:
            reader_anno = csv.reader(f_anno, delimiter=',', quotechar='"')
            reader_games = csv.reader(f_games, delimiter=',', quotechar='"')
            # skip headers
            next(reader_anno)
            next(reader_games)

            current_anno = next(reader_anno)

            for i, row in enumerate(reader_games):
                game_id = row[0]
                rotowire_game_id = int(row[3])
                sentences = sent_tokenize(row[4])

                logger.info(f"{game_id}")

                gid, game_data = games[rotowire_game_id]

                for j, sent in enumerate(sentences):
                    sent_tokens = sent.split()
                    sent_detok = self.detokenizer.detokenize(sent)
                    gt_texts = self.ss.fetch_text(sent_detok, game_data)

                    # print(sent)
                    # print(gt_texts)
                    # print("----------------")
                    gt_tokens = word_tokenize(gt_texts)

                    sep_token = [self.tokenizer.sep_token]
                    gt_labels = [Label.O.name for _ in range(len(gt_tokens)+1)] # also for the sep_token
                    sent_labels = [Label.O.name for _ in sent_tokens]

                    while True:
                        # process annotations until either the game or the current sentence changes
                        if game_id != current_anno[0][:4] or int(current_anno[1])-1 != j:
                            break

                        error_start = int(current_anno[4])
                        error_end = int(current_anno[5])
                        error_type = Label[current_anno[8]].name

                        for k in range(error_start, error_end+1):
                            sent_labels[k-1] = error_type

                        try:
                            current_anno = next(reader_anno)
                        except StopIteration:
                            break

                    # print(list(zip(sent_tokens, sent_labels)))
                    # print("====================")

                    tokens_all = gt_tokens + sep_token + sent_tokens
                    # a quote or backslash in the data may break the generated JSON
                    tokens_all = [str(token).replace('"', '\"').replace("\\","") for token in tokens_all]
                    labels_all = gt_labels + sent_labels

                    example = {
                        "text" : tokens_all,
                        "labels": labels_all
                    }

                    # first 10 games are in the test split for easier visual checks
                    if i < 10:
                        data["test"].append(example)
                    elif i < 55:
                        data["train"].append(example)
                    else:
                        data["dev"].append(example)


        for split in splits:
            out_fname = f"{split}.json"

            with open(os.path.join(self.args.data_dir, self.args.output, out_fname), "w") as f:
                s = json.dumps({"data" : data[split]}, ensure_ascii=False, indent=4, cls=CompactJSONEncoder)
                f.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=str, default="generated/simple_templates",
        help="Path to generated templates.")
    parser.add_argument("--rotowire_dir", type=str, default="../../datasets/rotowire",
        help="Path to data.")
    parser.add_argument("--data_dir", type=str, default="data",
        help="Path to data.")
    parser.add_argument("--output", type=str, required=True,
        help="Name of the output directory.")
    # parser.add_argument("--samples_per_game", type=int, default=10,
    #     help="Training samples generated per game.")
    parser.add_argument("--modification_rate", type=float, default=0.5,
        help="Proportion of modified entities.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed.")
    parser.add_argument("--start", type=int, default=None,
        help="Example to start from.")
    parser.add_argument("--end", type=int, default=None,
        help="Example to end at.")
    parser.add_argument("--split", type=str, required=True,
        help="Split: train / dev / test.")
    parser.add_argument("--annotations", type=str, default=None,
        help="Path to annotations for the games. If provided, training data will be generated using these annotations.")
    parser.add_argument("--games", default="games.csv", type=str,
        help="File with the games for the task.")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(os.path.join(args.data_dir, args.output), exist_ok=True)
    g = Generator(args)

    logger.info(f"Processing {args.split} data")
    games = load_games(args.templates, args.split)


    if args.annotations is not None:
        g.extract_annotated(games)
    else:
        g.generate(games, args.split)
