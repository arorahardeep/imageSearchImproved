# imageSearchImproved

This is the improvement over my imageSearch repo. 
- Under imageSearch the improvement was the introduction of bounding boxes to improve image search
- Under this repo, there is a further improvement by first captioning the images and then augmenting the image search above by adding searching based on caption similarity as well.

## Image Captioning

Quality of image caption is the determinant factor in getting this search to work, I did the following experiments

I used the code in repo - https://github.com/anuragmishracse/caption_generator to help caption the images. I ran this code on 
- Flick8k dataset - Here, I got poor results as there is no cat class hence my captions have no mention of cats
- PascalSentences dataset - http://vision.cs.uiuc.edu/pascal-sentences/ , I got somewhat better results but still not up to the mark

Then to prove the point this works, 
- I hand captioned 100 odd images and included the cat and dog breed names in the caption and it worked really well.
- I further realized I can use a model to classify these images into breeds by training on the following dataset
-- http://www.robots.ox.ac.uk/~vgg/data/pets/
- Then I can augment the PascalSentences dataset with the class labels of above model to match the performance on hand-captioned images.

## Combing the Bounding Box and Caption approach 

Finally I combined both the approaches by simple mean of similarity scores for both the approaches.

All the three approaches demo can been seen in the jupyter notebook "Search Image.ipynb"

## Files

The following class the take the idea of word similarity using word2vec and extends it to caption similarity by averaging the word vectors to make caption vectors

> python caption_similarity.py

The following file run in python 2.7 and prepares my images for caption generator repo code listed below

> python encode_img_for_cap.py

Run the python 2.7 code in the repo https://github.com/anuragmishracse/caption_generator to train your captioning model and then caption your images using it.

> caption_generator.py: The base script that contains functions for model creation, batch data generator etc.
> prepare_dataset.py: Prepares the dataset for training. Changes have to be done to this script if new dataset is to be used.
> train_model.py: Module for training the caption generator.
> test_model.py: Run it to caption your images 


This program creates caption similarity matrix, by comparing each caption with all others. This takes as input the caption generated in above step and outputs a csv file containing similarity scores for between all images x all images

> python img_cap_vecs.py

sample output as below
image_name,Cat_001.jpg,Cat_002.jpg,Cat_003.jpg
Cat_001.jpg,0.0,0.5622765421867371,0.591035008430481
Cat_002.jpg,0.5622765421867371,0.0,0.5671480298042297
Cat_003.jpg,0.591035008430481,0.5671480298042297,0.0


The following file as all the functions to perform image search

> python search.py

-- img_search_normal("Cat_008.jpg")   - Does a knive vec to vec similarity
-- img_search_frcnn("Cat_008.jpg")    - Improved image search using bounding boxes (Faster RNN)
-- img_search_caption("Cat_008.jpg")  - Further improvement using caption similarity (Above listed method)
-- img_search_combined("Cat_008.jpg") - Combination of bounding boxes and captions search.

unzip the imgdata.zip to get all images and run "Search Image.ipynb" to see the results.

