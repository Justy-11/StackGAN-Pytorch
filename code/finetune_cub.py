from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
import spacy
import random
import torchfile
from collections import Counter
from PIL import Image
from glob import glob
import os
import yaml
import numpy as np
import h5py
from txt2image_dataset import Text2ImageDataset
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import torch
import copy

from torchvision import transforms, datasets
from torchvision.models.inception import inception_v3
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from coreset import Coreset_Greedy
from cub2011 import Cub2011


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

images_path = config['birds_images_path']
embedding_path = config['birds_embedding_path']
text_path = config['birds_text_path']

train_classes = open(config['birds_train_split_path']).read().splitlines()


final_paths = []

class_paths = []
t = []

lengths = []

for _class in sorted(os.listdir(embedding_path)):
    if _class in train_classes:
        embeddings = []
        paths = []
        targets = []

        data_path = os.path.join(embedding_path, _class)
        txt_path = os.path.join(text_path, _class)
        for example, txt_file in zip(sorted(glob(data_path + "/*.pickle")), sorted(glob(txt_path + "/*.txt"))):
           
            with open(example, 'rb') as emb_f:
                example_data = pickle.load(emb_f)
    
            embeddings.append(np.mean(example_data, axis=0))
            
            paths.append(os.path.join(
                'birds/CUB_200_2011/images', example.split('/')[-2], example.split('/')[-1][:-3] + '.jpg'))
            # paths.append(os.path.join('birds/CUB_200_2011/images', _class, os.path.basename(example_file)[:-7] + '.jpg'))

            # embeddings.append(
            #     np.array(example_data[b'txt']).ravel())

        coreset = Coreset_Greedy(embeddings)
        temp = coreset.sample(0.25)

        class_paths += [paths[i] for i in temp]

final_paths += class_paths


paths_str = '\n'.join(final_paths)

with open(config['birds_coreset_path'], 'w') as file:
    file.write(paths_str)
