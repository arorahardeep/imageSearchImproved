#!/usr/bin/env python

"""
This program is a helper that loads all the images that need to be searched
"""


from keras.applications import inception_v3, resnet50
from keras.preprocessing import image
from keras.models import Model

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.spatial import distance
import pandas as pd

import numpy as np
import os
import glob
from os import walk

topn = 3

class ImgSearch:

    _model = None
    _search_path = './data2'
    _img_cols_px = 224
    _img_rows_px = 224
    _img_layers  = 3
    _img_features = None
    _search_scores = None
    _file_names = None
    _match_count = 0

    def __init__(self):
        #self._load_inception3()
        #self._load_resnet50()
        pass
        self.df = pd.read_csv("./data2/cap_cat_human.csv", index_col="image_name")

    def _load_inception3(self):
        base_model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
        #base_model.summary()
        layer_out = base_model.get_layer('avg_pool')
        self._model = Model(inputs=base_model.input, outputs=layer_out.output)

    def _load_resnet50(self):
        base_model = resnet50.ResNet50(weights='imagenet', include_top=True)
        #base_model.summary()
        layer_out = base_model.get_layer('avg_pool')
        self._model = Model(inputs=base_model.input, outputs=layer_out.output)

    def image_to_vector(self, image_name):
        img_path = os.path.join(self._search_path,image_name)
        img = image.load_img(img_path, target_size=( self._img_cols_px, self._img_rows_px))
        img = image.img_to_array(img)
        img = img.reshape(1, self._img_cols_px, self._img_rows_px, self._img_layers)
        preds = self._model.predict(img)
        return preds

    def create_img_vectors(self, location):
        f = []
        file_names = []
        for( dirpath, dirnames, filenames ) in walk(self._search_path):
            f.extend(filenames)
            for file in f:
                if file.endswith(".jpg"):
                    file_names.append(file.replace(".npy",""))
            break

        file_names = np.asarray(file_names)
        np.save(location + ".label", file_names)

        features = []
        for file_name in file_names:
            print(file_name)
            img_vec = self.image_to_vector(file_name)
            img_vecs = np.array(img_vec)
            features.append(img_vecs)

        features = np.asarray(features)
        np.save(location + ".npy", features)


    def show_image(self, name, label):
        img_path = os.path.join(self._search_path, name)
        img= mpimg.imread(img_path)
        plt.imshow(img)
        plt.title(label)
        plt.suptitle(name)
        plt.show()

    def load_imgs_vectors(self, location):
        if self._file_names is None:
            self._file_names = np.load(location + ".label.npy")
            self._img_features = np.load(location + ".npy")

    def img_search_similarity_matrix(self, csv_file):
        file_names = []
        features = []
        img_nos = 0
        for img_file in self._file_names:
            base_img_vec, indx = self.load_img_from_vec(img_file)
            similarity_vec = []
            vec_nos = 0
            for img_vec in self._img_features:
                if img_nos != vec_nos:
                    score_img = 1 - np.linalg.norm(img_vec - base_img_vec)/100
                    #score_img = 1 - distance.cosine(img_vec, base_img_vec)
                    score_cap = self.img_similarity_score_based_on_caption(self._file_names[vec_nos],self._file_names[img_nos])
                    score = 0.5*score_img + 0.5*score_cap
                    print(score_img, score_cap, score)
                else:
                    score = 0

                similarity_vec.append(score)
                vec_nos = vec_nos + 1

            file_names.append(img_file)
            features.append(similarity_vec)
            img_nos = img_nos + 1

        df = pd.DataFrame(features)
        df.columns = file_names
        df.index = file_names
        df.index.name = "image_name"
        df.to_csv(csv_file)

    def search_img(self, comparison_image):
        search_scores = []
        for feat in self._img_features :
            score = np.linalg.norm(feat - comparison_image)
            #score = 1 - distance.cosine(comparison_image, feat)
            search_scores.append(score)

        lowest = sorted(search_scores, key=float, reverse=False)
        self._search_scores = search_scores
        return lowest

    def increment_count(self,base_class, match_class):
        if self.get_class(base_class) == self.get_class(match_class):
            self._match_count = self._match_count + 1

    def show_img_with_score(self, score, base, flist):
        search_index = self._search_scores.index(score)
        fname = self._file_names[search_index]
        if fname not in flist and fname != base:
            flist.append(self._file_names[search_index])
            self.show_image(self._file_names[search_index], "Similarity Score : " + str(score))
            self.increment_count(base, self._file_names[search_index])
            print("Search Score = " + str(score))
        return flist

    def load_img_from_vec(self, image_name):
        fname =  self._file_names.tolist()
        indx = fname.index(image_name)
        return self._img_features[indx], indx

    def get_class(self, carname):
        return carname.split("_0")[0]

    def img_similarity_score_based_on_caption(self, img1, img2):
        return self.df.loc[img1][img2]


    def img_search_with_text(self, img_name):
        #df = pd.read_csv("./data2/cap_cat_human_1.csv", index_col="image_name")

        nlargest = topn
        order = np.argsort(-self.df.values, axis=1)[:, :nlargest]
        result = pd.DataFrame(self.df.columns[order],
                              columns=['top{}'.format(i) for i in range(1, nlargest+1)],
                              index=self.df.index)

        return result.ix[img_name]

    def get_img_caption_vector(self, img_name):
        df = pd.read_csv("./data2/cap_cat_human.csv", index_col="image_name")
        return df[img_name]


def imgs_to_npy():
    query = ImgSearch()
    query.create_img_vectors('./data2/cat_resnet50')


def img_search_common(search_img_name, img_vec_path):
    query = ImgSearch()
    query.load_imgs_vectors(img_vec_path)

    search_img,indx = query.load_img_from_vec(search_img_name)
    query.show_image(search_img_name, "Base Image")

    search_scores = query.search_img(search_img)
    print(len(search_scores))
    flist= []
    res = 1
    while (1):
        flist = query.show_img_with_score(search_scores[res], search_img_name, flist)
        if len(flist) >= topn:
            break
        res = res + 1


def img_search_frcnn(search_img_name):
    img_search_common(search_img_name, './data2/reg_cat_2')


def img_search_normal(search_img_name):
    img_search_common(search_img_name, './data2/cat_resnet50')


def img_search_caption(img_name):
    query = ImgSearch()
    query.show_image(img_name, "Base Image")
    results = query.img_search_with_text(img_name)
    for img_name in results:
        query.show_image(img_name, img_name)

def img_search_combined(img_name):
    query = ImgSearch()
    query.show_image(img_name, "Base Image")

    df = pd.read_csv("./data2/img_vec_mat.csv", index_col="image_name")
    nlargest = topn
    order = np.argsort(-df.values, axis=1)[:, :nlargest]
    result = pd.DataFrame(df.columns[order],
                              columns=['top{}'.format(i) for i in range(1, nlargest+1)],
                              index=df.index)

    for img_name in result.ix[img_name]:
        if img_name.count('.') > 1:
            img_name = img_name.rsplit('.',1)[0]
        query.show_image(img_name, img_name)


def main():
    #img_search_caption("Cat_001.jpg")
    ImgSrch = ImgSearch()
    #print(ImgSrch.img_similarity_score_based_on_caption("Cat_001.jpg","Cat_003.jpg"))
    ImgSrch.load_imgs_vectors('./data2/reg_cat_2')
    ImgSrch.img_search_similarity_matrix("./data2/img_vec_mat.csv")
    #print(ImgSrch.get_img_caption_vector('Cat_008.jpg'))
    #img_search_frcnn("Cat_008.jpg")


if __name__ == "__main__":
    main()

