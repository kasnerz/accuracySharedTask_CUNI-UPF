#!/usr/bin/env python

from sacremoses import MosesDetokenizer
import re

class Tokenizer:
    """
    Stub only, used for detokenization
    """
    def __init__(self):
        self.detokenizer = MosesDetokenizer(lang='en')

    def detokenize(self, s):
        tokens = s.split()
        return self.detokenizer.detokenize(tokens)


def normalize_tokens(tokens):
    # a quote or backslash in the data may break the generated JSON
    return [str(token).replace("\\", "").replace('"', r'\"') for token in tokens]