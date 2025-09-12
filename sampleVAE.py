import os
import numpy as np
import h5py
import scipy.io as sio
from myVAE import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

trainFile = 'data/VAE_dataset/ADCPMTrain.mat'
trainData = h5py.File(trainFile)
ADCPMTrain = np.array(trainData['ADCPMTrain'])
ADCPMTrain = ADCPMTrain.swapaxes(0, 2)
ADCPMTrain = np.expand_dims(ADCPMTrain, -1)
img_size = 128
latent_size = 8
latent_dim = latent_size ** 2
weight_res = 1
activation = "ELU"
model = VAEModel(latent_dim, activation, weight_res)
model.load_weights("ckpt_file/VAE64/model_1/vae.ckpt")  # Change to the saved model weight path
trainNum = ADCPMTrain.shape[0]
vae_lat_train = np.zeros((trainNum, latent_size, latent_size))
vae_rec_train = np.zeros((trainNum, img_size, img_size, 1))
batchN = 500
cnt = int(np.ceil(trainNum/batchN))
for i in range(cnt):
    z_mean_train, z_log_var_train = model.encoder(ADCPMTrain[i*batchN:(i+1)*batchN, :, :])
    z_sample_train = np.array(sampler(z_mean_train, z_log_var_train))
    vae_lat_train[i*batchN:(i+1)*batchN, :, :] = z_sample_train.reshape((z_sample_train.shape[0], latent_size, latent_size))
    vae_rec_train[i*batchN:(i+1)*batchN, :, :, :] = np.array(model.decoder(z_sample_train))
z_mean_train, z_log_var_train = model.encoder(ADCPMTrain[batchN*(cnt+1):trainNum, :, :])
z_sample_train = np.array(sampler(z_mean_train, z_log_var_train))
vae_lat_train[batchN*(cnt+1):trainNum, :, :] = z_sample_train.reshape((z_sample_train.shape[0], latent_size, latent_size))
vae_rec_train[batchN*(cnt+1):trainNum, :, :, :] = np.array(model.decoder(z_sample_train))
vae_rec_train = vae_rec_train.reshape((trainNum, img_size, img_size))
sio.savemat('data/ILDM_dataset/vae_lat_train.mat', {'vae_lat_train': vae_lat_train})
sio.savemat('data/ILDM_dataset/vae_rec_train.mat', {'vae_rec_train': vae_rec_train})

