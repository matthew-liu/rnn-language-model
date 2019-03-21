import numpy as np
import matplotlib.pyplot as plt
import utils


def plot_graphs(train_losses, test_losses, test_accuracies, graph_path, epoch=None):

    # cut off the fist epoch for better visualization
    if len(train_losses) > 1:
        train_losses = train_losses[1:]
        test_losses = test_losses[1:]
        test_accuracies = test_accuracies[1:]

    train_epochs = [t[0] for t in train_losses]
    train_losses = [t[1] for t in train_losses]

    test_epochs = [t[0] for t in test_losses]
    test_losses = [t[1] for t in test_losses]

    train_perplexities = np.exp(train_losses)
    test_perplexities = np.exp(test_losses)

    # loss
    plt.figure()
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_epochs, train_losses, 'b', label='train')
    plt.plot(test_epochs, test_losses, 'r', label='test')
    plt.legend()

    if epoch is not None:
        plt.savefig(graph_path + 'loss_%03d_epochs' % epoch)
    else:
        plt.savefig(graph_path + 'loss_cur')

    # perplexity
    plt.figure()
    plt.title('Perplexity Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.plot(train_epochs, train_perplexities, 'b', label='train')
    plt.plot(test_epochs, test_perplexities, 'r', label='test')
    plt.legend()

    if epoch is not None:
        plt.savefig(graph_path + 'perplex_%03d_epochs' % epoch)
    else:
        plt.savefig(graph_path + 'perplex_cur')

    test_epochs = [t[0] for t in test_accuracies]
    test_accuracies = [t[1] for t in test_accuracies]

    # accuracy
    plt.figure()
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.plot(test_epochs, test_accuracies, 'r', label='test')
    plt.legend()

    if epoch is not None:
        plt.savefig(graph_path + 'acc_%03d_epochs' % epoch)
    else:
        plt.savefig(graph_path + 'acc_cur')

    print('Final Test Accuracy:', test_accuracies[-1])
    print('Final Test Loss:', test_losses[-1])
    print('Final Test Perplexity:', test_perplexities[-1])


def plot_multi_losses(train_losses, test_losses, test_accs, labels, graph_path):

    # train loss
    assert len(train_losses) == len(labels)

    train_perpls = []
    test_perpls = []
    train_epochs = []
    test_epochs = []

    plt.figure()
    plt.title('Train Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    idx = 0
    for train_loss in train_losses:
        if len(train_loss) > 1:
            train_loss = train_loss[1:]

        epochs = [t[0] for t in train_loss]
        train_epochs.append(epochs)

        loss = [t[1] for t in train_loss]
        train_perpls.append(np.exp(loss))
        plt.plot(epochs, loss, label=labels[idx])
        idx += 1

    plt.legend()
    plt.savefig(graph_path + 'gross_train_loss_cur')

    # test loss
    plt.figure()
    plt.title('Test Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    idx = 0
    for test_loss in test_losses:
        if len(test_loss) > 1:
            test_loss = test_loss[1:]

        epochs = [t[0] for t in test_loss]
        test_epochs.append(epochs)

        loss = [t[1] for t in test_loss]
        test_perpls.append(np.exp(loss))

        plt.plot(epochs, loss, label=labels[idx])
        idx += 1

    plt.legend()
    plt.savefig(graph_path + 'gross_test_loss_cur')

    # train perplexity
    plt.figure()
    plt.title('Train Perplexity Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')

    for i in range(len(train_perpls)):
        plt.plot(train_epochs[i], train_perpls[i], label=labels[i])

    plt.legend()
    plt.savefig(graph_path + 'gross_train_perpl_cur')

    # test perplexity
    plt.figure()
    plt.title('Test Perplexity Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')

    for i in range(len(test_perpls)):
        plt.plot(test_epochs[i], test_perpls[i], label=labels[i])

    plt.legend()
    plt.savefig(graph_path + 'gross_test_perpl_cur')

    # test accuracy
    plt.figure()
    plt.title('Test Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')

    idx = 0
    for test_acc in test_accs:
        if len(test_acc) > 1:
            test_acc = test_acc[1:]

        epochs = [t[0] for t in test_acc]
        acc = [t[1] for t in test_acc]
        plt.plot(epochs, acc, label=labels[idx])
        idx += 1

    plt.legend()
    plt.savefig(graph_path + 'gross_test_acc_cur')


def main():
    dir_path = "./"
    log_path = dir_path + 'logs/m2/log.pkl'
    graph_path = dir_path + 'graphs/'

    # labels = ['w1', 'wd1', 'w2', 'wd2']
    # train_losses = [None, None, None, None]
    # test_losses = [None, None, None, None]
    # test_accs = [None, None, None, None]
    # train_losses[0], test_losses[0], test_accs[0] = utils.read_log('logs/m1_ndrop/log.pkl', ([], [], []))
    # train_losses[1], test_losses[1], test_accs[1] = utils.read_log('logs/m1/log.pkl', ([], [], []))
    # train_losses[2], test_losses[2], test_accs[2] = utils.read_log('logs/m2_ndrop/log.pkl', ([], [], []))
    # train_losses[3], test_losses[3], test_accs[3] = utils.read_log('logs/m2/log.pkl', ([], [], []))

    labels = ['w1', 'w2', 'w3']
    train_losses = [None, None, None]
    test_losses = [None, None, None]
    test_accs = [None, None, None]
    train_losses[0], test_losses[0], test_accs[0] = utils.read_log('logs/m1_ndrop/log.pkl', ([], [], []))
    train_losses[1], test_losses[1], test_accs[1] = utils.read_log('logs/m2_ndrop/log.pkl', ([], [], []))
    train_losses[2], test_losses[2], test_accs[2] = utils.read_log('logs/m3/log.pkl', ([], [], []))
    train_losses[3], test_losses[3], test_accs[3] = utils.read_log('logs/m2/log.pkl', ([], [], []))

    plot_multi_losses(train_losses, test_losses, test_accs, labels, graph_path)
    # train_losses, test_losses, test_accuracies = utils.read_log(log_path, ([], [], []))
    # plot_graphs(train_losses, test_losses, test_accuracies, graph_path)


if __name__ == '__main__':
    main()

