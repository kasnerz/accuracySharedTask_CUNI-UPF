#!/usr/bin/env python3

import sys
import os
from sentence_transformers import SentenceTransformer, util

class SentenceScorer:
    """
    Use sentence embeddings and cosine similarity to retrieve relevant
    sentences out of all sentences generated for a game. 
    """
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v2')
        # self.model.cuda()

    def score(self, texts, sentence):
        emb_texts = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        emb_sent = self.model.encode(sentence, convert_to_tensor=True, show_progress_bar=False)
        
        # efficient N-to-1 computation
        emb_sent = emb_sent.repeat(len(texts)).view(len(texts),-1)

        scores = util.pytorch_cos_sim(emb_texts, emb_sent).diag()
        scored_sentences = list(sorted(zip(texts, scores), key=lambda x: x[1], reverse=True))

        return scored_sentences

    def retrieve_ctx_scored(self, sentence, game_data, cnt):
        """
        Returns a list of tuples (sentence, sim_score)
        """
        texts = game_data.get_all()
        scored_sentences = self.score(sentence=sentence, texts=texts)

        return scored_sentences[:cnt]

    def retrieve_ctx(self, sentence, game_data, cnt):
        """
        Returns a paragraph with context (concatenated relevant sentences)
        """
        scored_sentences = self.retrieve_ctx_scored(sentence, game_data, cnt)
        text = " ".join([x[0] for x in scored_sentences])

        return text