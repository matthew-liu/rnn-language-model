import torch
import pickle
import re
import torch.utils.data


# word vocab
def prepare_data_word(input_folder_path, output_folder_path, unk_threshold=1, test_split=0.1):

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
        if words[word] >= unk_threshold:
            voc2ind[word] = len(voc2ind)

    print('word vocab size:', len(voc2ind))

    # transform the data into an integer representation of the tokens.
    tokens = [voc2ind[word] if word in voc2ind else voc2ind[unknown] for word in data]

    ind2voc = {val: key for key, val in voc2ind.items()}

    train_set_length = int((1 - test_split) * len(tokens))

    train_text = tokens[:train_set_length]
    test_text = tokens[train_set_length:]

    pickle.dump({'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind': voc2ind},
                open(output_folder_path + 'train.pkl', 'wb'))
    pickle.dump({'tokens': test_text, 'ind2voc': ind2voc, 'voc2ind': voc2ind},
                open(output_folder_path + 'test.pkl', 'wb'))


def prepare_data_char(input_folder_path, output_folder_path, test_split=0.1):
    # load all train/dev/test data into one word vector
    data = []
    with open(input_folder_path + 'movie.train') as f:
        data += re.sub('\s+', ' ', f.read()).strip()  # fix spacing
    with open(input_folder_path + 'movie.dev') as f:
        data += re.sub('\s+', ' ', f.read()).strip()  # fix spacing
    with open(input_folder_path + 'movie.test') as f:
        data += re.sub('\s+', ' ', f.read()).strip()  # fix spacing

    print('number of chars:', len(data))

    # Compute voc2ind
    voc2ind = {}
    for char in data:
        if char not in voc2ind:
            voc2ind[char] = len(voc2ind)

    print('char vocab size:', len(voc2ind))

    # transform the data into an integer representation of the tokens.
    tokens = [voc2ind[char] for char in data]

    ind2voc = {val: key for key, val in voc2ind.items()}

    train_set_length = int((1 - test_split) * len(tokens))

    train_text = tokens[:train_set_length]
    test_text = tokens[train_set_length:]

    pickle.dump({'tokens': train_text, 'ind2voc': ind2voc, 'voc2ind': voc2ind},
                open(output_folder_path + 'train.pkl', 'wb'))
    pickle.dump({'tokens': test_text, 'ind2voc': ind2voc, 'voc2ind': voc2ind},
                open(output_folder_path + 'test.pkl', 'wb'))


class WordVocab(object):
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

    # Returns the size of the WordVocab.
    def __len__(self):
        return len(self.voc2ind)


class CharVocab(object):
    def __init__(self, data_file):
        with open(data_file, 'rb') as data_file:
            dataset = pickle.load(data_file)
        self.ind2voc = dataset['ind2voc']
        self.voc2ind = dataset['voc2ind']

    def idx_to_char(self, idx):
        return self.ind2voc[int(idx)]

    # Returns a string representation of the tokens.
    def array_to_words(self, arr):
        return ''.join([self.ind2voc[int(ind)] for ind in arr])

    # Returns a torch tensor representing each token in words.
    def words_to_array(self, words):
        return torch.LongTensor([self.voc2ind[word] for word in words])

    # Returns the size of the vocabulary.
    def __len__(self):
        return len(self.voc2ind)


# splitting the data set into N chunks where N is the batch_size and
# the chunks are contiguous parts of the data.
# For each batch, we return one sequence from each of the chunks.
class ContinuousDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, sequence_length, batch_size, char_vocab=True):
        super(ContinuousDataset, self).__init__()

        self.sequence_length = sequence_length
        self.batch_size = batch_size

        if char_vocab:
            self.vocab = CharVocab(data_file)
        else:
            self.vocab = WordVocab(data_file)

        with open(data_file, 'rb') as data_pkl:
            dataset = pickle.load(data_pkl)

        tokens = dataset['tokens']

        # removing extra bits at the end of tokens
        extra = len(tokens) % batch_size
        tokens = tokens[:len(tokens) - extra]

        # divide tokens into N=batch_size chunks
        self.chunks = {}
        chunk_length = int(len(tokens) / batch_size)
        assert (chunk_length >= sequence_length + 1)
        extra = (chunk_length - 1) % sequence_length
        chunk_size = int((chunk_length - 1) / sequence_length)
        for i in range(batch_size):
            chunk_tokens = tokens[i * chunk_length:(i + 1) * chunk_length]
            self.chunks[i] = []
            for j in range(chunk_size):
                self.chunks[i].append(chunk_tokens[j * sequence_length:(j + 1) * sequence_length + 1])
            if extra > 0:
                self.chunks[i].append(chunk_tokens[chunk_length - extra - 1:chunk_length])

        # total number of unique sequences
        if extra > 0:
            chunk_size += 1
        self.size = len(self.chunks) * chunk_size
        assert (self.size % batch_size == 0)

    def __len__(self):
        # return the number of unique sequences
        return self.size

    def __getitem__(self, idx):
        # Return the data and label for a word sequence as torch long tensors.
        # You should return a single entry for the batch using the idx to decide which chunk you are
        # in and how far down in the chunk you are.

        batch_idx = int(idx / self.batch_size)
        batch_offset = idx % self.batch_size

        data = torch.tensor(self.chunks[batch_offset][batch_idx]).long()
        return data[:-1], data[1:]

    def vocab_size(self):
        return len(self.vocab)


DATA_PATH = './data/movie/'
PROCESSED_DATA_PATH = './data/movie/'
UNK_THRESHOLD = 5


def main():
    prepare_data_word(DATA_PATH, PROCESSED_DATA_PATH, UNK_THRESHOLD)
    # prepare_data_char(DATA_PATH, PROCESSED_DATA_PATH)


if __name__ == "__main__":
    main()
