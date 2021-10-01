import pickle


# 模型压缩部分
class Compressor(object):
    def __init__(self, model, voter_mask, voting_mask):
        self.voter_mask = voter_mask
        self.voting_mask = voting_mask
        self.model = model

    def tune(self, data_train, data_test, batch_size, n_step):
        self.model.fit_with_mask(data_train, data_test, batch_size=batch_size, n_step=n_step)

    def save(self, path):
        self.model.save_model(path)
        with open('%s/masks.pkl', 'wb') as f:
            pickle.dump({'voter_mask': self.voter_mask, 'voting_mask': self.voting_mask})
