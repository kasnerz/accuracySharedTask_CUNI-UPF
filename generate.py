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
from utils.utils import Label
from utils.tokenizer import Tokenizer as Detokenizer
from utils.tokenizer import normalize_tokens
from utils.sentence_scorer import SentenceScorer
from utils.utils import random_choice_neq

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
        self.ents_total_cnt = 0
        self.ents_total_mod_cnt = 0
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


    def modify_numbers(self, orig_text, mode="cardinal"):
        """
        Modify numbers (integers) based on the mode (cardinal / ordinal). Gaussian noise with a cut-off at
        zero is applied until the number is different from the original number.

        If `orig_text` contains non-digit numbers, alpha2digit is used to transform all the numbers 
        in text to digit-only variants and transformed back to text after modifications.
        """
        orig_text = str(orig_text)
        text = alpha2digit(orig_text, "en")

        if mode == "ordinal":
            # alpha2digit cannot handle "first", "second" and "third", has to be handled separately
            extra_ordinals = {"first" : "1st", "second" : "2nd", "third" : "3rd"}
            for k, v in extra_ordinals.items():
                # replace only if surrounded by word breaks to avoid things like seconds -> 2nds
                text = re.sub(rf"\b{k}\b", rf"{v}", text)

        to_lex = (orig_text != text)    # some numbers in lexical form were transformed to digits
        chars = [ch for ch in text]

        # find all number-like groups
        numbers = [(m.group(0), m.start(0), m.end(0)) for m in re.finditer(r"\d+\w*", text)]

        for num_str, start, end in numbers:
            # extract digits for modifications
            num = int(re.search(r"\d+", num_str).group(0))
            modified_num = num

            # apply Gaussian noise with sigma=3 until the number differs
            # negative numbers do not occur in texts -> threshold at zero
            while modified_num == num:
                modified_num = max(0, int(random.gauss(num, 3)))

            if to_lex:
                modified_num = num2words(modified_num, to=mode)

            chars[start:end] = [ch for ch in str(modified_num)]

        return "".join(chars)


    def modify_text_with_numbers(self, ent_tokens, mode):
        # can be a multi-word expression such as "88% to 77%" or "39 minutes"
        tokens = []
        labels = []
        for i, token in enumerate(ent_tokens):

            token = str(token)

            # this enables to modify some of the numbers in cases such as if "14-11" is identified as a single entity
            if i > 0 and random.random() > self.args.modification_rate:
                modified_token = token
            else:    
                modified_token = self.modify_numbers(token, mode=mode)

            if modified_token != token:
                tokens.append(modified_token)
                labels.append(Label.NUMBER)
            else:
                tokens.append(token)
                labels.append(Label.O)

        return tokens, labels


    def modify_entity(self, doc, game_data, entity):
        """
        Make the entity errorneous.

        doc - output of Spacy NER NLP pipeline
        game_data - GameData object for the game
        entity - (text, label, start, end, list_of_tokens) as identified Spacy NER
        """
        ent_text = entity[0]
        ent_type = entity[1]
        ent_tokens = entity[4]

        if ent_text in self.days_of_week:
            # change a day of week to a random day
            random_day = random_choice_neq(l=self.days_of_week, neq_to=ent_text)
            tokens = [random_day]
            labels = [Label.NAME]
        elif ent_type == "PERSON":
            # swap a player with another player present in the game
            random_player = random_choice_neq(l=list(game_data.get_players()), neq_to=ent_text)
            tokens = word_tokenize(random_player)
            labels = [Label.NAME for token in tokens]
        elif ent_type == "GPE":
            # GPE - usually a name of the city; select another city from all the cities
            # in the Rotowire dataset
            random_city = random_choice_neq(l=self.cities, neq_to=ent_text)
            tokens = word_tokenize(random_city)
            labels = [Label.NAME for token in tokens]
        elif ent_type == "CARDINAL" \
                or ent_type == "PERCENT" \
                or ent_type == "TIME" \
                or ent_type == "QUANTITY" \
                or re.match(r"\d+\s*-\s*\d+", ent_text):
            tokens, labels = self.modify_text_with_numbers(ent_tokens, mode="cardinal")
        elif ent_type == "ORDINAL" \
            or "half" in ent_text \
            or "quarter" in ent_text:   # "X-th quarter/half" (may be recognized as date), X->Y
            tokens, labels = self.modify_text_with_numbers(ent_tokens, mode="ordinal")
        elif ent_text in game_data.get_teams():
            # change to the other team name
            random_team = random_choice_neq(l=list(game_data.get_teams()), neq_to=ent_text)
            tokens = word_tokenize(random_team)
            labels = [Label.NAME for token in tokens]
        elif ent_type in ["NORP", "PRODUCT", "EVENT", "DATE", "LOC", "ORG", "FAC"]:
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

        self.ents_total_cnt += len(ne)

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
                self.ents_total_mod_cnt += 1
                current_entity_started = False

        tokens = [str(t) for t in tokens]

        return (tokens, labels)

    def generate(self, games, split, ctx_sizes):
        data = {ctx_size : [] for ctx_size in ctx_sizes}

        start = self.args.start or 0
        end = self.args.end or len(games)

        for game_id in range(start, end):
            if game_id >= len(games):
                break

            game_data = games[game_id]

            if not game_data:
                continue

            rotowire_game = self.rotowire[split][game_id]
            rotowire_summary = rotowire_game["summary"]
            rotowire_summary_sents = sent_tokenize(" ".join(rotowire_summary))

            for sent in rotowire_summary_sents:
                sent_detok = self.detokenizer.detokenize(sent)

                # run the Spacy NLP pipeline for identifying named entities
                doc = self.spacy_nlp(sent_detok)
                modif_tokens, modif_labels = self.augment(doc, game_data)
                modif_sent_detok = self.detokenizer.detokenize(" ".join(modif_tokens))

                # retrieve N scored sentences where N is the largest context size
                ctx_scored = self.ss.retrieve_ctx_scored(modif_sent_detok, game_data, cnt=sorted(ctx_sizes)[-1])

                for ctx_size in ctx_sizes:
                    ctx = " ".join([x[0] for x in ctx_scored[:ctx_size]])
                    ctx_tokens = word_tokenize(ctx)

                    # there is not "%" sign in the task data
                    # "\" may break the resulting JSON
                    ctx_tokens = [str(token).replace("%", "percent").replace("\\","") for token in ctx_tokens]
                    modif_tokens = [str(token).replace("%", "percent").replace("\\","") for token in modif_tokens]

                    example = {
                        "ctx" : ctx_tokens,
                        "sent" : modif_tokens,
                        "labels": modif_labels
                    }
                    data[ctx_size].append(example)

                if len(data) > 0 and len(data[ctx_sizes[0]]) % 20 == 0:
                    logger.info(f"{len(data)} examples generated (game id {game_id}).")

        out_fname = f"{split}.json"
        logger.info(f"{self.ents_total_mod_cnt}/{self.ents_total_cnt} entities modified.")

        if self.args.start or self.args.end:
            # partial file for parallelization, will be merged
            out_fname = f"{split}-{self.args.start:04d}-{self.args.end:04d}.json"

        for ctx_size in ctx_sizes:
            with open(os.path.join(self.args.data_dir, self.args.output + f"_ctx{ctx_size}", out_fname), "w") as f:
                s = json.dumps({"data" : data[ctx_size]}, ensure_ascii=False, indent=4, cls=CompactJSONEncoder)
                f.write(s)


    def extract_annotated(self, games, ctx_sizes):
        """
        Generate training data from the annotated data for the task.
        """
        splits = ["train", "dev", "test"]
        data = {
            split : {ctx_size : [] for ctx_size in ctx_sizes}
                for split in splits
        }
        # the games from Rotowire are used for retrieving the context for the annotated errors
        with open(self.args.annotations) as f_anno, open(self.args.games) as f_games:
            reader_anno = csv.reader(f_anno, delimiter=',', quotechar='"')
            reader_games = csv.reader(f_games, delimiter=',', quotechar='"')
            # skip headers
            next(reader_anno)
            next(reader_games)

            current_anno = next(reader_anno)

            for i, row in enumerate(reader_games):
                if i < 10:
                    split = "test"
                elif i < 55:
                    split = "train"
                else:
                    split = "dev"

                game_id = row[0]
                rotowire_game_id = int(row[3])
                sentences = sent_tokenize(row[4])

                logger.info(f"Processing game {game_id}")
                game_data = games[rotowire_game_id]

                for j, sent in enumerate(sentences):
                    sent_tokens = normalize_tokens(sent.split())
                    sent_detok = self.detokenizer.detokenize(sent)
                    sent_labels = [Label.O.name for _ in sent_tokens]

                    while True:
                        # process annotations until either the game id or the sentence id changes
                        if game_id != current_anno[0][:4] or int(current_anno[1])-1 != j:
                            break

                        error_start = int(current_anno[4])
                        error_end = int(current_anno[5])
                        error_type = Label[current_anno[8]].name

                        # replace OK labels by error labels for all relevant tokens
                        for k in range(error_start, error_end+1):
                            sent_labels[k-1] = error_type

                        try:
                            current_anno = next(reader_anno)
                        except StopIteration:
                            break

                    ctx_scored = self.ss.retrieve_ctx_scored(sent_detok, game_data, cnt=sorted(ctx_sizes)[-1])

                    for ctx_size in ctx_sizes:
                        ctx = " ".join([x[0] for x in ctx_scored[:ctx_size]])
                        ctx_tokens = normalize_tokens(word_tokenize(ctx))
                        
                        example = {
                            "ctx" : ctx_tokens,
                            "sent" : sent_tokens,
                            "labels": sent_labels
                        }
                        if i % 5 == 0 and j == 0:
                            logger.info(example)

                        data[split][ctx_size].append(example)
                        
        for split in splits:
            out_fname = f"{split}.json"

            for ctx_size in ctx_sizes:
                with open(os.path.join(self.args.data_dir, self.args.output + f"_ctx{ctx_size}", out_fname), "w") as f:
                    s = json.dumps({"data" : data[split][ctx_size]}, ensure_ascii=False, indent=4, cls=CompactJSONEncoder)
                    f.write(s)



    def get_stats_ents(self):
        stats = defaultdict(int)

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

                logger.info(f"Processing game {game_id}")

                for j, sent in enumerate(sentences):
                    sent_detok = self.detokenizer.detokenize(sent)
                    doc = self.spacy_nlp(sent_detok)
                    nes = [(ent.text, 
                           ent.label_, 
                           ent.start, 
                           ent.end, 
                           [doc[i] for i in range(ent.start, ent.end)]) for ent in doc.ents]

                    for ne in nes:
                        if ne[1] in ["ORG", "FAC", "DATE", "PERSON", "GPE"]:
                            stats["name_ents"] += 1
                        elif ne[1] in ["CARDINAL", "ORDINAL"]:
                            stats["number_ents"] += 1

                    while True:
                        # process annotations until either the game id or the sentence id changes
                        if game_id != current_anno[0][:4] or int(current_anno[1])-1 != j:
                            break
                        error_type = Label[current_anno[8]].name
                        stats[error_type] += 1
                        try:
                            current_anno = next(reader_anno)
                        except StopIteration:
                            break

        print(stats)
        name_ents, number_ents, name, number = [stats[key] for key in ["name_ents", "number_ents", "NAME", "NUMBER"]]
        print(f"Ratio NAME: {name}/{name_ents} ({name/name_ents})")
        print(f"Ratio NUMBER: {number}/{number_ents} ({number/number_ents})")
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", type=str, required=True,
        help="Type of templates (simple / compact).")
    parser.add_argument("--rotowire_dir", type=str, default="../../datasets/rotowire",
        help="Path to data.")
    parser.add_argument("--data_dir", type=str, default="data",
        help="Path to data.")
    parser.add_argument("--output", type=str, required=True,
        help="Name of the output directory.")
    parser.add_argument("--ctx", type=int, default=None,
        help="Number of sentences retrieved for the context.")
    parser.add_argument("--modification_rate", type=float, default=0.5,
        help="Proportion of modified entities.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed.")
    parser.add_argument("--start", type=int, default=None,
        help="Example to start from.")
    parser.add_argument("--end", type=int, default=None,
        help="Example to end at.")
    parser.add_argument("--split", type=str, default=None,
        help="Split: train / dev / test.")
    parser.add_argument("--annotations", type=str, default=None,
        help="Path to annotations for the games. If provided, training data will be generated using these annotations.")
    parser.add_argument("--games", default="games.csv", type=str,
        help="File with the games for the task.")
    parser.add_argument("--get_stats_ents", action="store_true",
        help="Only compute statistics for entities.")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ctx_sizes = [args.ctx] if args.ctx is not None else [5, 10, 20, 40]

    for ctx_size in ctx_sizes:
        os.makedirs(os.path.join(args.data_dir, args.output + f"_ctx{ctx_size}"), exist_ok=True)

    templates_path = f"./context/{args.templates}"
    g = Generator(args)

    if args.get_stats_ents:
        g.get_stats_ents()
    else:
        if args.annotations is not None:
            games = load_games(templates_path, "test")
            g.extract_annotated(games, ctx_sizes)
        else:
            logger.info(f"Processing {args.split} data")
            games = load_games(templates_path, args.split)
            g.generate(games, args.split, ctx_sizes)
