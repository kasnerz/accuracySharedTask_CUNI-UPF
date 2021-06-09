#!/usr/bin/env python3
import random

def random_choice_neq(l, neq_to):
    """
    Return a random element from a list. The element must be different from the given element `neq_to`.
    """
    assert any([el != neq_to for el in l]), f"Cannot select element not equal to {neq_to} in {l}"

    while True:
        choice = random.choice(l)
        if choice != neq_to:
            return choice