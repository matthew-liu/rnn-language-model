import utils
import matplotlib.pyplot as plt


def plot_graphs(log_path):
    train_losses, test_losses, test_accuracies = utils.read_log(log_path, ([], [], []))

    # cut off the fist epoch for better visualization
    del train_losses[0]
    del test_losses[0]
    del test_accuracies[0]

    print(train_losses)
    print(test_losses)
    print(test_accuracies)

    train_epochs = [t[0] for t in train_losses]
    train_losses = [t[1] for t in train_losses]

    test_epochs = [t[0] for t in test_losses]
    test_losses = [t[1] for t in test_losses]

    train_perplexities = np.exp(train_losses)
    test_perplexities = np.exp(test_losses)

    # plot the loss, perlexity & accuracy
    plt.figure()
    plt.title('Train loss (blue) vs. Test loss (red)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_epochs, train_losses, 'b')
    plt.plot(test_epochs, test_losses, 'r')
    plt.savefig(GRAPH_PATH + 'loss_%03d_epochs' % epoch)

    plt.figure()
    plt.title('Train Perplexity (blue) vs. Test Perplexity (red)')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.plot(train_epochs, train_perplexities, 'b')
    plt.plot(test_epochs, test_perplexities, 'r')
    plt.savefig(GRAPH_PATH + 'perplexity_%03d_epochs' % epoch)

    test_epochs = [t[0] for t in test_accuracies]
    test_accuracies = [t[1] for t in test_accuracies]

    plt.figure()
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.plot(test_epochs, test_accuracies, 'r')
    plt.savefig(GRAPH_PATH + 'accuracy_%03d_epochs' % epoch)