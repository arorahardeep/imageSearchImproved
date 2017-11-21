#!/usr/bin/env python

'''
This program measures similarity between two captions using word2vec embeddings
@author : Hardeep Arora
@date	: 14-Nov-2017
'''

import numpy as np
import math
from nltk.corpus import stopwords
from gensim.models import KeyedVectors


class CaptionVector:
    def __init__(self, caption, embeddings):
        self.embedding_model = embeddings
        self.vector = self.caption_to_vector(caption)

    def caption_vec_based_on_avg_vecs(self, vec_set, ignore = []):
        if len(ignore) == 0:
            return np.mean(vec_set, axis = 0)
        else:
            return np.dot(np.transpose(vec_set), ignore) / sum(ignore)

    def caption_to_vector(self, caption):
        cached_stop_words = stopwords.words("english")
        caption = caption.lower()
        words_in_caption = [word for word in caption.split() if word not in cached_stop_words]
        vec_set = []
        for word in words_in_caption:
            try:
                word_vecs=self.embedding_model[word]
                vec_set.append(word_vecs)
            except:
                pass
        return self.caption_vec_based_on_avg_vecs(vec_set)

    def calc_cosine_sim(self, other_caption_vec):
        cosine_similarity = np.dot(self.vector, other_caption_vec) / (np.linalg.norm(self.vector) * np.linalg.norm(other_caption_vec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity=0
        except:
            cosine_similarity=0
        return cosine_similarity


class CaptionSimilarity:
    word_embeddings = 'embeddings/GoogleNews-vectors-negative300.bin'

    def __init__(self):
        print("Loading embeddings vector...")
        self.embedding_model = KeyedVectors.load_word2vec_format(self.word_embeddings, binary=True)

    def caption_similarity(self, caption1, caption2):
        cap_vec_1 = CaptionVector(caption1, self.embedding_model)
        cap_vec_2 = CaptionVector(caption2, self.embedding_model)
        similarityScore = cap_vec_1.calc_cosine_sim(cap_vec_2.vector)
        return similarityScore


def main():
    cs = CaptionSimilarity()
    cap1 = "The brown cat with furry hair"
    cap2 = "The cat in brown color is sitting on table"
    print(cs.caption_similarity(cap1, cap2))

    cap1 = "The brown cat with furry hair"
    cap2 = "The dog in brown color is sitting on table"
    print(cs.caption_similarity(cap1, cap2))


if __name__ == "__main__":
    main()