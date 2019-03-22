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

import preprocess as prep
import models

import sys


BEAM_WIDTH = 10


def predict_completion(model, input, prob, hidden, vocab):
    completion = vocab.idx_to_char(input)
    stop_char = ' '
    log_prob = torch.log(prob).item()

    model.eval()
    with torch.no_grad():

        output, hidden = model.inference(input, hidden)

        while True:
            # sample = torch.argmax(output)
            prob, sample = torch.max(output, dim=0)
            log_prob += torch.log(prob).item()
            next_char = vocab.idx_to_char(sample.item())
            completion += next_char

            if len(completion) > 2 and next_char == stop_char:
                return log_prob, completion

            output, hidden = model.inference(sample, hidden)


def predict_completions(model, device, seed_words, vocab, n=8, width=30):

    assert len(seed_words) > 0

    model.eval()
    with torch.no_grad():
        seed_words_arr = vocab.words_to_array(seed_words)

        # Computes the initial hidden state from the prompt (seed words).
        output, hidden = None, None
        for ind in seed_words_arr:
            data = ind.to(device)
            output, hidden = model.inference(data, hidden)

        # sample n=width next idices
        probs, indices = torch.sort(output, descending=True)
        probs = probs[:width]
        sample = indices[:width]

        results = [predict_completion(model, sample[i], probs[i], hidden, vocab) for i in range(len(sample))]
        results.sort(key=lambda tup: tup[0], reverse=True)
        return [word for log_prob, word in results[:n]]


def predict_next_word_char(model, device, seed_words, vocab, n=8, beam_width=12):
    model.eval()
    stop_char = ' '
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

        while True:
                new_beams = []
                # loop through old beams list to create new beams list
                for beam in beams:
                    seq, output, hidden, score = beam
                    if len(seq) > 0:
                        last_char = vocab.idx_to_char(seq[-1])
                        if last_char == stop_char:
                            new_beams.append(beam)
                            continue

                    _, indices = torch.sort(output, descending=True)
                    samples = indices[:beam_width]
                    # samples = torch.multinomial(output, beam_width, replacement=True)

                    for sample in samples:
                        next_score = score + np.log(output[sample].item())
                        # next_score -= np.log(len(seq) + 1)
                        next_output, next_hidden = model.inference(sample, hidden)
                        new_beam = (seq + [sample.item()], next_output, next_hidden, next_score)
                        new_beams.append(new_beam)

                if len(new_beams) == len(beams):
                    beams = new_beams[:n]
                    break
                new_beams.sort(key=lambda tup: tup[3], reverse=True)
                beams = new_beams[:beam_width]

        # outputs = [(beam[3], vocab.array_to_words(beam[0])) for beam in beams]
        # print(outputs)
        # return [word for prob, word in outputs]

        return [vocab.array_to_words(beam[0]) for beam in beams]


def predict_sequence(model, device, seed_words, sequence_length, vocab, sampling_strategy='max',
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


def predict_next_word(model, device, seed_words, vocab, n=8):
    model.eval()
    with torch.no_grad():
        seed_words_arr = vocab.words_to_array(seed_words)

        # Computes the initial hidden state from the prompt (seed words).
        hidden = None
        for ind in seed_words_arr:
            data = ind.to(device)
            output, hidden = model.inference(data, hidden)

        _, indices = torch.sort(output, descending=True)
        return vocab.array_to_words(indices[:n]).split(' ')


def main():

    model_path = './best_models/char_xlarge'
    feature_size = 650
    char_vocab = True

    if char_vocab:
        vocab = prep.CharVocab(prep.PROCESSED_DATA_PATH + 'train.pkl')
    else:
        vocab = prep.WordVocab(prep.PROCESSED_DATA_PATH + 'train.pkl')

    print('vocab size:', len(vocab))

    USE_CUDA = False
    use_cuda = USE_CUDA and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using device', device)

    model = models.ForwardGRUNet(len(vocab), feature_size).to(device)
    model.load_last_model(model_path)

    seed_words = ''

    while True:
        seed_words = sys.stdin.readline().strip()
        if seed_words == 'exit':
            exit()
        # print('seed_words:', seed_words)
        if len(seed_words) > 0:

            if char_vocab:
                # completions = predict_completions(model, device, seed_words + ' ', vocab, n=8)
                completions = predict_next_word_char(model, device, seed_words + ' ', vocab, n=8)
            else:
                completions = predict_next_word(model, device, seed_words, vocab, n=8)
            print('next-word prediction:\t', completions)

    seed_words = 'It\'s such a nice day today'
    sequence_length = 8
    #
    # generated_sentence = predict_sequence(model, device, seed_words, sequence_length, vocab, 'max')
    # print('generated with max\t', generated_sentence)
    #
    # for ii in range(5):
    #     generated_sentence = predict_sequence(model, device, seed_words, sequence_length, vocab, 'sample')
    #     print('generated with sample\t', generated_sentence)
    #
    for ii in range(5):
        generated_sentence = predict_sequence(model, device, seed_words, sequence_length, vocab, 'beam')
        print('generated with beam\t', generated_sentence)


if __name__ == '__main__':
    main()
