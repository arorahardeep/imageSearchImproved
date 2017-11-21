#!/usr/bin/env python

'''
This module prepares the images for captioning.
'''

import pandas as pd
import pickle as pickle
from keras.preprocessing import image
from vgg16 import VGG16
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

counter = 0

def load_image(path):
    img = image.load_img(path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return np.asarray(x)


def load_encoding_model():
    model = VGG16(weights='imagenet', include_top=True, input_shape = (224, 224, 3))
    return model


def get_encoding(model, img):
    global counter
    counter += 1
    image = load_image('./data2/'+str(img))
    pred = model.predict(image)
    pred = np.reshape(pred, pred.shape[1])
    print "Encoding image: "+str(counter)
    print pred.shape
    return pred


def get_img_list(img_region_path):
    imgs_df = pd.read_csv(img_region_path)
    file_names = []
    for i in range(0,imgs_df.shape[0]):
        file_names.append(imgs_df['image_name'].iloc[i])
    return file_names


def encode_imgs(test_imgs):
    encoded_images = {}
    encoding_model = load_encoding_model()
    for img in test_imgs:
        encoded_images[img] = get_encoding(encoding_model, img)
    with open( "encoded_images.p", "wb" ) as pickle_f:
        pickle.dump( encoded_images, pickle_f )


def main():
    tst_imgs = get_img_list('./data2/img_regions_cats_1.csv')
    encode_imgs(tst_imgs)

if __name__ == "__main__":
    main()