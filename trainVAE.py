import os
import numpy as np
import h5py
from scipy import io
from myVAE import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MyCallBack, self).__init__()
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('total_loss')
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.model.save_weights(f'ckpt_file/VAE64/model_{epoch + 1}/vae.ckpt')


trainFile = 'data/VAE_dataset/ADCPMTrain.mat'
trainData = h5py.File(trainFile)
ADCPMTrain = np.array(trainData['ADCPMTrain'])
ADCPMTrain = ADCPMTrain.swapaxes(0, 2)
ADCPMTrain = np.expand_dims(ADCPMTrain, -1)
img_size = 128
latent_size = 8
latent_dim = latent_size ** 2
batch_size = 512
shuffle_buffer = 3000
epochs = 3000
weight_res = 10
activation = "ELU"
train_dataset = (
        tf.data.Dataset.from_tensor_slices(ADCPMTrain)
        .shuffle(buffer_size=shuffle_buffer)
        .batch(batch_size)
    )
model = VAEModel(latent_dim, activation, weight_res)
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    run_eagerly=True
)
checkpoint_callback1 = MyCallBack()
history = model.fit(
    train_dataset,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint_callback1],
)
reconstruction_loss = history.history['reconstruction_loss']
io.savemat('results/reconstruction_loss.mat', {'reconstruction_loss': reconstruction_loss})
kl_loss = history.history['kl_loss']
io.savemat('results/kl_loss.mat', {'kl_loss': kl_loss})
total_loss = history.history['total_loss']
io.savemat('results/total_loss.mat', {'total_loss': total_loss})