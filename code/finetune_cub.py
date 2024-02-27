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
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.cuda.manual_seed_all(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# train_transform = transforms.Compose([
#     transforms.Resize(299),
#     transforms.CenterCrop(299),
#     transforms.RandomRotation(45),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# test_transform = transforms.Compose([
#     transforms.Resize(299),
#     transforms.CenterCrop(299),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])


# train_data = Cub2011(
#     'birds', train=True, download=False, transform=train_transform)
# val_data = Cub2011(
#     'birds', train=False, download=False, transform=test_transform)

# # train_data = ConcatDataset([train_data, test_data])

# new_train_data = train_data
# new_val_data = val_data

# train_loader = DataLoader(new_train_data, batch_size=128,
#                           shuffle=True, drop_last=True)
# val_loader = DataLoader(new_val_data, batch_size=128, shuffle=False)

# model = inception_v3(weights='DEFAULT')

# # for param in model.parameters():
# #     param.requires_grad = False

# model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, 200)
# model.fc = torch.nn.Linear(model.fc.in_features, 200)

# device = 'cuda:1'

# model.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criteria = torch.nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


# best_model = copy.deepcopy(model)
# best_acc = 0

# patience = 3
# early_stopping_counter = 0

# epochs = 20

# for epoch in range(epochs):
#     model.train()

#     train_loss, val_loss = 0, 0
#     train_acc, val_acc = 0, 0

#     for inputs, targets in train_loader:
#         inputs = inputs.to(device)
#         targets = targets.to(device)

#         optimizer.zero_grad()

#         outputs, aux_outputs = model(inputs)

#         loss = criteria(outputs, targets) + criteria(aux_outputs, targets)

#         train_loss += loss.item() * inputs.size(0)

#         _, preds = torch.max(outputs, 1)
#         train_acc += torch.sum(preds == targets)

#         loss.backward()
#         optimizer.step()

#     scheduler.step()

#     model.eval()

#     for inputs, targets in val_loader:
#         inputs = inputs.to(device)
#         targets = targets.to(device)

#         with torch.no_grad():
#             outputs = model(inputs)

#         loss = criteria(outputs, targets)

#         val_loss += loss.item() * inputs.size(0)

#         _, preds = torch.max(outputs, 1)
#         val_acc += torch.sum(preds == targets)

#     val_acc = val_acc.item()/len(new_val_data)

#     print('Epoch: ', epoch+1, 'Train Acc:', train_acc.item()/len(new_train_data),
#           'Train Loss:', train_loss /
#           len(new_train_data), 'Val Acc:', val_acc,
#           'Train Loss:', val_loss/len(new_val_data), )

#     if val_acc > best_acc:
#         best_acc = val_acc
#         best_model = copy.deepcopy(model)
#         early_stopping_counter = 0
#     else:
#         early_stopping_counter += 1

#     if early_stopping_counter >= patience:
#         break

# torch.save(best_model, 'birds_inception_v311.pth')


# best_model = torch.load('flower_best_model.pth').to(device)
# best_model.fc = torch.nn.Identity()
# best_model.eval()


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

# vectorizer = KeyphraseCountVectorizer(stop_words='english')
# kw = KeyBERT()

for _class in sorted(os.listdir(embedding_path)):
    if _class in train_classes:
        embeddings = []
        paths = []
        targets = []

        data_path = os.path.join(embedding_path, _class)
        txt_path = os.path.join(text_path, _class)
        for example, txt_file in zip(sorted(glob(data_path + "/*.pickle")), sorted(glob(txt_path + "/*.txt"))):
            # for example in sorted(glob(data_path + "/*.t7")):

            # f = open(txt_file, "r")
            # txt = f.readlines()
            # f.close()

            # doc = kw.extract_keywords(
            #     ' '.join(txt), stop_words='english', keyphrase_ngram_range=(1, 1), top_n=50, use_mmr=True, diversity=0.8)

            # lengths.append(len(doc))

            # if len(doc) <= 35:
            # if True:
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
