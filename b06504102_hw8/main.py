import numpy as np
import random
import torch

from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,TensorDataset)
import torchvision.transforms as transforms

from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.optim import Adam, AdamW

from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.vq import vq, kmeans

# from qqdm import qqdm, format_str
from tqdm import tqdm
import pandas as pd

import pdb  # use pdb.set_trace() to set breakpoints for debugging

from Mydataset import CustomTensorDataset
from Mynet import fcn_autoencoder, conv_autoencoder, VAE, loss_vae, Resnet, DCGAN

train = np.load('data-bin/trainingset.npy', allow_pickle=True)
test = np.load('data-bin/testingset.npy', allow_pickle=True)
print(train.shape)
print(test.shape)

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Training hyperparameters
same_seeds(19530615)
num_epochs = 1024
batch_size = 256 # medium: smaller batchsize
learning_rate = 1e-3

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'DCGAN'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'resnet': Resnet(), 'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE(), 'DCGAN':DCGAN() }
model = model_classes[model_type].cuda()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=16, eps=1e-12 ,verbose=True)

best_loss = np.inf
model.train()

# # qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
# # for epoch in qqdm_train:
# for epoch in range(num_epochs):
#     tot_loss = list()
#     for data in tqdm(train_dataloader):

#         # ===================loading=====================
#         if model_type in ['cnn', 'vae', 'resnet', 'DCGAN']:
#             img = data.float().cuda()
#         elif model_type in ['fcn']:
#             img = data.float().cuda()
#             img = img.view(img.shape[0], -1)

#         # ===================forward=====================
#         output = model(img)
#         if model_type in ['vae']:
#             loss = loss_vae(output[0], img, output[1], output[2], criterion)
#         else:
#             loss = criterion(output, img)

#         tot_loss.append(loss.item())
#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # ===================save_best====================
#     mean_loss = np.mean(tot_loss)
#     if mean_loss < best_loss:
#         best_loss = mean_loss
#         torch.save(model, 'best_model_{}.pt'.format(model_type))
#         print(f'Best loss:{best_loss}')

#     # ===================log========================
#     # qqdm_train.set_infos({
#     #   'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
#     #   'loss': f'{mean_loss:.4f}',
#     # })
#     print(f'Epoch:{epoch} | Mean loss:{mean_loss}')
#     # ===================save_last========================
#     torch.save(model, 'last_model_{}.pt'.format(model_type))

#     scheduler.step(best_loss)

#########################################
############### Inference ##A#############
#########################################

eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = 'best_model_DCGAN.pt'
model = torch.load(checkpoint_path)
model.eval()

# prediction file 
out_file = 'PREDICTION_FILE.csv'

    
anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader): 
        if model_type in ['cnn', 'vae', 'resnet', 'DCGAN']:
            img = data.float().cuda()
        elif model_type in ['fcn']:
            img = data.float().cuda()
            img = img.view(img.shape[0], -1)
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type in ['cnn', 'resnet', 'fcn', 'DCGAN']:
            output = output
        elif model_type in ['res_vae']:
            output = output[0]
        elif model_type in ['vae']: # , 'vqvae'
            output = output[0]
        if model_type in ['fcn']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['Predicted'])
df.to_csv(out_file, index_label = 'Id')

