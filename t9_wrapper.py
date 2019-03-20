import preprocess as prep
import models
import torch


class GRUModel:
    def __init__(self, model_path, feature_size, config, char_vocab=False):
        super(GRUModel, self).__init__()

        if char_vocab:
            self.vocab = prep.CharVocab(prep.PROCESSED_DATA_PATH + 'train.pkl')
        else:
            self.vocab = prep.WordVocab(prep.PROCESSED_DATA_PATH + 'train.pkl')

        print('vocab size:', len(self.vocab))

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.gpu else "cpu")
        print('Using device', self.device)

        self.model = models.ForwardGRUNet(len(self.vocab), feature_size).to(self.device)
        self.model.load_last_model(model_path)

        self.unk = '[???]'

    def get_predict(self, context):
        self.model.eval()
        with torch.no_grad():
            seed_words_arr = self.vocab.words_to_array(context)

            # Computes the initial hidden state from the prompt (seed words).
            output, hidden = None, None
            for ind in seed_words_arr:
                data = ind.to(self.device)
                output, hidden = self.model.inference(data, hidden)

            return output.detach().cpu().numpy()

    def id_to_token(self, i):
        if i == -1:
            return self.unk
        return self.vocab.ind2voc[i]

    def token_to_id(self, token):
        if token in self.vocab.voc2ind:
            return self.vocab.voc2ind[token]
        return -1
