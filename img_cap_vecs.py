#!/usr/bin/env python

"""
This program creates caption similarity matrix, by comparing each caption with all others.
@author : Hardeep Arora
@date   : 14 Nov 2017

"""

import numpy as np
import pandas as pd
from caption_similarity import CaptionSimilarity


class ImgCap2Vecs:

    def __init__(self):
        self.cs = CaptionSimilarity()
        self.cd = {}

    def create_cap_dict(self, caption_file):
        imgs_df = pd.read_csv(caption_file)
        for i in range(0,imgs_df.shape[0]):
            self.cd[imgs_df['image_name'].iloc[i]] = imgs_df['caption'].iloc[i]

    def img_caption_similarity_matrix(self, caption_file, csv_file):
        self.create_cap_dict(caption_file)
        file_names = []
        features = []
        for key_base,value_base in self.cd.items():
            similarity_vec = []
            for key_comp,value_comp in self.cd.items():
                if key_base != key_comp:
                    similarity_vec.append(self.cs.caption_similarity(value_base, value_comp))
                else:
                    similarity_vec.append(0)

            file_names.append(key_base)
            features.append(similarity_vec)

        df = pd.DataFrame(features)
        df.columns = file_names
        df.index = file_names
        df.index.name = "image_name"
        df.to_csv(csv_file)


def main():
    icv = ImgCap2Vecs()
    icv.img_caption_similarity_matrix("./data2/img_cap_human.txt","./data2/cap_cat_human.csv")


if __name__ == "__main__":
    main()