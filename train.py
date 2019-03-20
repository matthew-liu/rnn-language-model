import os
import torch
import multiprocessing
import numpy as np
import torch.optim as optim
import utils

import models
import preprocess as prep
import plot


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def train(model, device, train_loader, lr, epoch, log_interval):
    model.train()
    losses = []
    hidden = None

    # update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        # Separates the hidden state across batches.
        # Otherwise the backward would try to go all the way to the beginning every time,
        # leading to vanishing gradients!
        if hidden is not None:
            hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden = model(data, hidden)
        # pred = output.max(-1)[1]
        loss = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return np.mean(losses)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        hidden = None
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output, hidden = model(data, hidden)
            test_loss += model.loss(output, label, reduction='mean').item()
            pred = output.max(-1)[1]
            correct_mask = pred.eq(label.view_as(pred))
            num_correct = correct_mask.sum().item()
            correct += num_correct
            # Comment this out to avoid printing test results
            if batch_idx % 10 == 0:
                print('Input\t%s\nLabel\t%s\nPred\t%s\n\n' % (
                    train_loader.dataset.vocab.array_to_words(data[0]),
                    train_loader.dataset.vocab.array_to_words(label[0]),
                    train_loader.dataset.vocab.array_to_words(pred[0])))

    test_loss /= len(test_loader)
    test_acc = 100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) * test_loader.dataset.sequence_length,
        100. * correct / (len(test_loader.dataset) * test_loader.dataset.sequence_length)))
    return test_loss, test_acc


# Word Vocab
SEQUENCE_LENGTH = 8
BATCH_SIZE = 256
FEATURE_SIZE = 650
TEST_BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005

# Char Vocab
# SEQUENCE_LENGTH = 60
# BATCH_SIZE = 256
# FEATURE_SIZE = 512
# TEST_BATCH_SIZE = 256
# EPOCHS = 50
# LEARNING_RATE = 0.02
# WEIGHT_DECAY = 0.0005

USE_CUDA = True
PRINT_INTERVAL = 10

DIR_PATH = "./"
LOG_PATH = DIR_PATH + 'logs/log.pkl'
BEST_MODEL_PATH = DIR_PATH + 'best_models/'
if not os.path.exists(BEST_MODEL_PATH):
    os.makedirs(BEST_MODEL_PATH)

GRAPH_PATH = DIR_PATH + 'graphs/'
if not os.path.exists(GRAPH_PATH):
    os.makedirs(GRAPH_PATH)

char_vocab = False

data_train = prep.ContinuousDataset(prep.PROCESSED_DATA_PATH + 'train.pkl', SEQUENCE_LENGTH, BATCH_SIZE, char_vocab)
data_test = prep.ContinuousDataset(prep.PROCESSED_DATA_PATH + 'test.pkl', SEQUENCE_LENGTH, TEST_BATCH_SIZE, char_vocab)
vocab = data_train.vocab
print('vocab size:', len(vocab))

use_cuda = USE_CUDA and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Using device', device)

num_workers = multiprocessing.cpu_count()
print('num workers:', num_workers)

kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)

model = models.ForwardGRUNet(data_train.vocab_size(), FEATURE_SIZE).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

start_epoch = model.load_last_model(DIR_PATH + 'checkpoints')

if start_epoch < EPOCHS:

    train_losses, test_losses, test_accuracies = utils.read_log(LOG_PATH, ([], [], []))

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            lr = LEARNING_RATE * np.power(0.25, (int(epoch / 10)))
            train_loss = train(model, device, train_loader, lr, epoch, PRINT_INTERVAL)
            test_loss, test_accuracy = test(model, device, test_loader)
            train_losses.append((epoch, train_loss))
            test_losses.append((epoch, test_loss))
            test_accuracies.append((epoch, test_accuracy))
            utils.write_log(LOG_PATH, (train_losses, test_losses, test_accuracies))
            model.save_best_model(test_accuracy, BEST_MODEL_PATH + '%05f.pt' % test_accuracy)
            plot.plot_graphs(train_losses, test_losses, test_accuracies, GRAPH_PATH)

    except KeyboardInterrupt as ke:
        print('Interrupted')
    except:
        import traceback
        traceback.print_exc()
    finally:
        print('Saving final model')
        model.save_model(DIR_PATH + 'checkpoints/%03d.pt' % epoch, 0)
        print('train loss:', train_losses[:2])
        plot.plot_graphs(train_losses, test_losses, test_accuracies, GRAPH_PATH, epoch)
