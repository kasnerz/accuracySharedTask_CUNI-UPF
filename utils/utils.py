#!/usr/bin/env python3
import random
from enum import Enum

class Label(Enum):
    """
    Label which corresponds to annotation tags as defined by the task
    """
    O = 0           # OK
    NUMBER = 1      
    NAME = 2        
    WORD = 3       
    NOT_CHECKABLE = 4   
    CONTEXT = 5   
    OTHER = 6   

    @classmethod
    def label2id(cls):
        return {label.name : label.value for label in cls}

    @classmethod
    def id2label(cls):
        return {label.value : label.name for label in cls}


def random_choice_neq(l, neq_to):
    """
    Return a random element from a list. The element is different from the given element `neq_to`.
    """
    assert any([el != neq_to for el in l]), f"Cannot select element not equal to {neq_to} in {l}"

    while True:
        choice = random.choice(l)
        if choice != neq_to:
            return choice