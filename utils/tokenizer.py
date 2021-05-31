#!/usr/bin/env python

from sacremoses import MosesDetokenizer
from nltk import word_tokenize, sent_tokenize
import re

class Tokenizer:
    def __init__(self):
        self.detokenizer = MosesDetokenizer(lang='en')

    def detokenize(self, s):
        tokens = s.split()

        return self.detokenizer.detokenize(tokens)