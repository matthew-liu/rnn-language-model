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

BEAM_WIDTH = 10


def generate_language(model, device, seed_words, sequence_length, vocab, sampling_strategy='max',
                      beam_width=BEAM_WIDTH):
    model.eval()

    with torch.no_grad():
        seed_words_arr = vocab.words_to_array(seed_words)

        # Computes the initial hidden state from the prompt (seed words).
        hidden = None
        for ind in seed_words_arr:
            data = ind.to(device)
            output, hidden = model.inference(data, hidden)

        outputs = []
        # Initializes the beam list.
        beams = [([], output, hidden, 0)]

        for ii in range(sequence_length):

            if sampling_strategy == 'max':
                sample = torch.argmax(output)
                output, hidden = model.inference(sample, hidden)
                outputs.append(sample.item())

            elif sampling_strategy == 'sample':
                sample = torch.multinomial(output, 1)
                output, hidden = model.inference(sample, hidden)
                outputs.append(sample.item())

            elif sampling_strategy == 'beam':
                new_beams = []
                # loop through old beams list to create new beams list
                for beam in beams:
                    seq, output, hidden, score = beam
                    samples = torch.multinomial(output, beam_width, replacement=True)
                    for sample in samples:
                        next_score = score + np.log(output[sample].item())
                        next_output, next_hidden = model.inference(sample, hidden)
                        new_beam = (seq + [sample.item()], next_output, next_hidden, next_score)
                        new_beams.append(new_beam)
                new_beams.sort(key=lambda tup: tup[3], reverse=True)
                beams = new_beams[:beam_width]

        if sampling_strategy == 'beam':
            outputs = beams[0][0]

        return vocab.array_to_words(seed_words_arr.tolist() + outputs)