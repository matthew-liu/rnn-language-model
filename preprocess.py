import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import sys
import pickle
import re
import utils
import matplotlib.pyplot as plt


# word vocab
def prepare_data(input_folder_path, output_folder_path, unk_threshold=1,   v=0.1):

    # load all train/dev/test data into one word vector
    data = []
    with open(input_folder_path + 'movie.train') as f:
        data += re.sub('\s+', ' ', f.read()).strip().split(' ')  # fix spacing
    with open(input_folder_path + 'movie.dev') as f:
        data += re.sub('\s+', ' ', f.read()).strip().split(' ')  # fix spacing
    with open(input_folder_path + 'movie.test') as f:
        data += re.sub('\s+', ' ', f.read()).strip().split(' ')  # fix spacing

    # compute word frequency
    words = {}
    for word in data:
        if word not in words:
            words[word] = 1
        else:
            words[word] += 1

    # Compute voc2ind
    voc2ind = {}
    unknown = '[???]'
    voc2ind[unknown] = 0

    for word in words:
        if words[word] > unk_threshold:
            voc2ind[word] = len(voc2ind)

    print('vocab size:', len(voc2ind))

    # transform the data into an integer representation of the tokens.
    tokens = [voc2ind[word] if word in voc2ind else voc2ind[unknown] for word in data]

    ind2voc = {val: key for key, val in voc2ind.items()}

    num = int(0.9 * len(tokens))

    train_text = tokens[:num]
    test_text = tokens[num:]

    pickle.dump({'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind': voc2ind},
                open(output_folder_path + 'train.pkl', 'wb'))
    pickle.dump({'tokens': test_text, 'ind2voc': ind2voc, 'voc2ind': voc2ind},
                open(output_folder_path + 'test.pkl', 'wb'))


class Vocabulary(object):
    def __init__(self, data_file):
        with open(data_file, 'rb') as data_file:
            dataset = pickle.load(data_file)
        self.ind2voc = dataset['ind2voc']
        self.voc2ind = dataset['voc2ind']

    # Returns a string representation of the tokens.
    def array_to_words(self, arr):
        return ' '.join([self.ind2voc[int(ind)] for ind in arr])

    # Returns a torch tensor representing each token in words.
    def words_to_array(self, words):
        return torch.LongTensor([self.voc2ind[word] for word in words.split(' ')])

    # Returns the size of the vocabulary.
    def __len__(self):
        return len(self.voc2ind)


def main():
    data_path = './data/movie/'
    processed_data_path = './data/movie/'
    unk_threshold = 2

    prepare_data(data_path, processed_data_path, unk_threshold)


if __name__ == "__main__":
    main()
