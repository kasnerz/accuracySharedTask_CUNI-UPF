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
        emb_sent = emb_sent.repeat(len(texts)).view(len(texts),-1)

        scores = util.pytorch_cos_sim(emb_texts, emb_sent).diag()
        scored_sentences = list(sorted(zip(texts, scores), key=lambda x: x[1], reverse=True))

        return scored_sentences

    def retrieve_ctx_scored(self, sentence, game_data, cnt):
        texts = game_data.get_all()
        scored_sentences = self.score(sentence=sentence, texts=texts)

        return scored_sentences[:cnt]

    def retrieve_ctx(self, sentence, game_data, cnt=15):
        scored_sentences = retrieve_ctx_scored(sentence, game_data, cnt)

        # for i,(txt, score) in enumerate(scored_sentences[:30]):
        #     print(f"{i}\t{float(score):.3f}\t{txt}")

        text = " ".join([x[0] for x in scored_sentences])

        # print(sentence)
        # print("-----------")
        # print(text)
        # print("===========")

        return text