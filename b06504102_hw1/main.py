'''
Modified:
1. loss function: MSE -> RMSE
2. feature: 4 症狀、shop、public_transit、worried_finance、陽性
3. batch size: 32
4. earlystop: 1024
5. optimizer: Adam(lr=0.0001, wd=0.01)
6. Normalize: disable
7. n_epoch: 15000
8. model: [input->16]->[16->16]->[16->output]
'''

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from COVID19Dataset import COVID19Dataset
from MyNet import NeuralNet
# For data preprocess
import numpy as np
import os
import csv
# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    # print('train: ',(loss_record['train'])[-1])
    # print('dev: ',(loss_record['dev'])[-1])
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    # print(sum((targets-preds)**2)/len(targets))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


def prep_dataloader(path, mode, batch_size, n_jobs=0):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
        
        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            # print(model.cal_loss(pred, y))
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if(epoch%10 == 0):
            loss_record['train'].append(mse_loss.detach().cpu().item())
            loss_record['dev'].append(dev_mse)
        epoch += 1

        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

def getConfig():
    config = {
        'n_epochs': 15000,                # maximum number of epochs
        'batch_size': 32,               # mini-batch size for dataloader
        'optimizer': 'Adam',              # optimization algorithm (optimizer in torch.optim)
        'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
            'lr': 0.0001,                 # learning rate of SGD
            # 'momentum': 0.9,              # momentum for SGD
            'weight_decay': 0.01,
        },
        'early_stop': 1024,               # early stopping epochs (the number epochs since your model's last improvement)
        'save_path': 'models/model.pth'  # your model will be saved here
    }
    return config

if __name__ == "__main__" :
    tr_path = 'covid.train.csv'  # path to training data
    tt_path = 'covid.test.csv'   # path to testing data
    config = getConfig()
    device = get_device()                 # get the current available device ('cpu' or 'cuda')
    os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/

    myseed = 0  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

    tr_set = prep_dataloader(tr_path, 'train', config['batch_size'])
    dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'])
    tt_set = prep_dataloader(tt_path, 'test', config['batch_size'])
    model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device

    model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)

    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
    model.load_state_dict(ckpt)
    # plot_pred(dv_set, model, device)  # Show prediction on the validation set
    # plot_learning_curve(model_loss_record, title='deep model')
    preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
    save_pred(preds, 'pred.csv')         # save prediction file to pred.csv


'''
Reference:

1. https://scikit-learn.org/stable/modules/feature_selection.html

'''