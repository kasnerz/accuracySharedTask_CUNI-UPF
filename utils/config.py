#!/usr/bin/env python3

from enum import Enum

class Label(Enum):
    O = 0           # OK
    NUMBER = 1      # number error

    @classmethod
    def label2id(cls):
        return {label.name : label.value for label in cls}

    @classmethod
    def id2label(cls):
        return {label.value : label.name for label in cls}