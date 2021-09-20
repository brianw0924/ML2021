'''
Reference: 

0. Label Smoothing: https://blog.csdn.net/Najlepszy/article/details/100540130
1. CrossEntropy Weight: 
    2.0 https://stackoverflow.com/questions/61414065/pytorch-weight-in-cross-entropy-loss
    2.1 https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731
    2.2 https://discuss.pytorch.org/t/weights-in-weighted-loss-nn-crossentropyloss/69514
2. L-softmax: https://github.com/amirhfarzaneh/lsoftmax-pytorch
3. Angular-softmax: 
    4.0 https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
    4.1 https://github.com/topics/angular-softmax

'''

import numpy as np
import torch
import torch.nn as nn
import gc
from Utilities import get_device, same_seeds
from sklearn import feature_selection as fs
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from Mynet import Classifier
# from Mynet_ import Classifier,AngularPenaltySMLoss
from TIMITDataset import TIMITDataset
from LabelSmoothing import LabelSmoothingLoss

# training parameters
same_seeds(1126)
num_epoch = 2048                         # number of training epoch
learning_rate = 0.01                     # learning rate
Feature_select = 'type_2'                # 1~4
Validation_split = 'all_train'           # ratio, shuffle, all_train
CrossEntropyLossWeight = None            # CrossEntropyLossWeight
Normalization = None                     # Normalization
Criterion = 'LabelSmoothingLoss'         # CrossEntropyLoss, AngularSoftMaxLoss, LabelSmoothingLoss
Scheduler_type = 'ReduceLR'              # CosineAnnealingLR, ReduceLROnPlateau, ExponentialLR
BATCH_SIZE = 8192                        # Batch size
earlystop = 2048                         # Early Stop epoch
model_path = './model_window_1.ckpt'     # which window of tmux session
saving_path = 'pred_41.csv'              # saving name


# Load data & split validation set
print('Loading data ...')
data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')
print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))

if Feature_select == 'type_1':
    # Feature Selection - 1: KBest
    kbest = fs.SelectKBest(fs.f_classif,k=20*11)
    train = kbest.fit_transform(train,train_label)
    indice = kbest.get_support()
    test  = test[:,indice]
elif Feature_select == 'type_2':
    # Feature Selection - 2: 前7後7
    train_1_to_7  = train[:,39*0:39*7]
    train_5_to_11 = train[:,39*4:39*11] 
    train = np.concatenate((train_1_to_7,train_5_to_11),axis=1)
    del train_1_to_7, train_5_to_11
    test_1_to_7   = test[:,39*0:39*7]
    test_5_to_11  = test[:,39*4:39*11]
    test  = np.concatenate((test_1_to_7,test_5_to_11),axis=1)
    del test_1_to_7, test_5_to_11
elif Feature_select == 'type_3':
    # Feature Selection - 3: 1,3,5,6,7,9,11
    train = np.concatenate((train[:,39*0:39*1],train[:,39*2:39*3],train[:,39*4:39*7],train[:,39*8:39*9],train[:,39*10:39*11]),axis=1)
    test  = np.concatenate((test[:,39*0:39*1],test[:,39*2:39*3],test[:,39*4:39*7],test[:,39*8:39*9],test[:,39*10:39*11]),axis=1)
elif Feature_select == 'type_4':
    # Feature Selection - 4: 去頭尾2個, 中間兩倍
    train_3_to_6  = train[:,39*2:39*6]
    train_6_to_9 = train[:,39*5:39*9] 
    train = np.concatenate((train_3_to_6,train_6_to_9),axis=1)
    del train_3_to_6, train_6_to_9
    test_3_to_6   = test[:,39*2:39*6]
    test_6_to_9  = test[:,39*5:39*9]
    test  = np.concatenate((test_3_to_6,test_6_to_9),axis=1)
    del test_3_to_6, test_6_to_9

if Validation_split == 'ratio':
    # 直接切後面
    VAL_RATIO = 0.1
    percent = int(train.shape[0] * (1 - VAL_RATIO))
    train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
elif Validation_split == 'shuffle':
    # Shuffle
    train_x, val_x, train_y, val_y = train_test_split(train, train_label, test_size=0.2, random_state=0)
elif Validation_split == 'all_train':
    # 全部丟進去 train
    train_x, train_y, val_x, val_y = train, train_label, train, train_label
    

if CrossEntropyLossWeight != None:
    # crossentropy weight
    label_weight = torch.zeros(39)
    for i in range(len(train_y)):
        label_weight[int(train_y[i])]+=1
    label_weight = torch.FloatTensor(label_weight)

    s = label_weight.sum()
    label_weight = 1 - (label_weight / s)

    print("CrossEntropyWeight: ",label_weight)

if Normalization != None:
    # Normalize
    mean  = train_x.mean(axis=0)
    std   = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    val_x   = (val_x - mean) / std
    test  = (test - mean) / std
    del mean, std

print('Size of train set: {}'.format(train_x.shape))
print('Size of train_label set: {}'.format(train_y.shape))
print('Size of validation set: {}'.format(val_x.shape))
print('Size of validation_label set: {}'.format(val_y.shape))

# get device 
device = get_device()
print(f'DEVICE: {device}')

# Dataloader
train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# create model, define a loss function, and optimizer
model = Classifier().to(device)


# CRITERION
if Criterion == 'AngularSoftMaxLoss':
    criterion = AngularPenaltySMLoss(loss_type='Sphereface').to(device)
elif Criterion == 'CrossEntropyLoss':
    if CrossEntropyLossWeight != None:
        criterion = nn.CrossEntropyLoss(weight=label_weight).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)
elif Criterion == 'LabelSmoothingLoss':
    criterion = LabelSmoothingLoss(classes=39, smoothing= 0.1).to(device)

# OPTIMIZER
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# SCHEDULER
if Scheduler_type == 'ExpLR':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
elif Scheduler_type == 'ReduceLR':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, verbose=True)
elif Scheduler_type == 'CosineLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)

# start training
best_acc = 0.0
last_update = 0
for epoch in range(num_epoch):
    # print(optimizer.param_group[0]['lr'])
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()
    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                last_update = 0
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
            else:
                last_update+=1
                
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))
    scheduler.step(val_acc)
    # scheduler.step()
    if(last_update == earlystop):
        print("Early stop at: ",epoch+1)
        break

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')


# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))

# Predict
predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

# outcome
with open(saving_path, 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))

print("Finish saving: ",saving_path)
print("Model: ",model_path)