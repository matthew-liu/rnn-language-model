# rnn-language-model

### self-trained word embedding:

#### - model 1 (small):
unk_threshold: 5

vocab_size: 29002

feature_size (hidden_dim): 128

num_gru_layer: 2

__best test accuracy: 15.52%__

__best test accuracy dropout(0.1): 15.36%__

#### - model 2:

unk_threshold: 5

vocab_size: 29002

feature_size (hidden_dim): 300

num_gru_layer: 2

__best test accuracy: 16.03%__

__best test accuracy dropout(0.1): 15.87%__

### self-trained char embedding:

#### model 3:
number of chars: 16896309

char vocab_size: 92