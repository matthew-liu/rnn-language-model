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
    print('Final Test Perplexity:', test_perplexities[-1])


def main():
    dir_path = "./"
    log_path = dir_path + 'logs/log.pkl'
    graph_path = dir_path + 'graphs/'

    train_losses, test_losses, test_accuracies = utils.read_log(log_path, ([], [], []))
    plot_graphs(train_losses, test_losses, test_accuracies, graph_path)


if __name__ == '__main__':
    main()

