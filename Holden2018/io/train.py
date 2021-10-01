import keras as K
import numpy as np
import os
class Schedule:
    """
    The schedule of the training processing.
    """

    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.1:
            return self.initial_lr * (0.8 ** 1)
        elif epoch_idx < self.epochs * 0.2:
            return self.initial_lr * (0.8 ** 2)
        elif epoch_idx < self.epochs * 0.3:
            return self.initial_lr * (0.8 ** 3)
        elif epoch_idx < self.epochs * 0.4:
            return self.initial_lr * (0.8 ** 4)
        elif epoch_idx < self.epochs * 0.5:
            return self.initial_lr * (0.8 ** 5)
        elif epoch_idx < self.epochs * 0.6:
            return self.initial_lr * (0.8 ** 6)
        return self.initial_lr * (0.8 ** 7)


class LossHistory(K.callbacks.Callback):
    """
    Config the loss history
    """

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))


class CustomModelCheckpoint(K.callbacks.Callback):
    """
    The schedule of saving model.
    """

    def __init__(self, model, path):
        super().__init__()

        self.model = model
        self.path = path

        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['val_loss']

        if loss < self.best_loss:
            path = self.path.format(epoch=epoch + 1, **logs)
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, loss))
            self.model.save_weights(path, overwrite=True)
            self.model.save_weights(os.path.join('models', 'Holden.hdf5'), overwrite=True)
            self.best_loss = loss