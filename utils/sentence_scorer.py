#!/usr/bin/env python3

import sys
import os
from sentence_transformers import SentenceTransformer, util

class SentenceScorer:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        self.model.cuda()

    def score(self, texts, sentence):
        #Compute embedding for both lists

        emb_texts = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        emb_sent = self.model.encode(sentence, convert_to_tensor=True, show_progress_bar=False)

        scores = []

        for emb_text in emb_texts:
            score = util.pytorch_cos_sim(emb_text, emb_sent)
            scores.append(score)

        scored_sentences = list(sorted(zip(texts, scores), key=lambda x: x[1], reverse=True))

        return scored_sentences

