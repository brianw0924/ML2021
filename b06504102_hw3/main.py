'''
Reference: 

0. Model: 
    0. https://medium.com/雞雞與兔兔的工程世界/機器學習-ml-note-cnn演化史-alexnet-vgg-inception-resnet-keras-coding-668f74879306
1. Data augmentation: 
    0. https://chih-sheng-huang821.medium.com/03-pytorch-dataaug-a712a7a7f55e
    1. https://blog.csdn.net/u014380165/article/details/79167753
 
'''






import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
# from torchvision.datasets import DatasetFolder
from MyDatasetFolder import DatasetFolder
from tqdm import tqdm
from Mynet import Classifier_0, Classifier_1
# from SemiSupervised import get_pseudo_labels
import torchvision.models as models

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Parameters
same_seeds(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
n_epochs = 4096
earlystop = 4096
threshold = 0.99
do_semi = False
Scheduler = "Cosine"
model_path = "model_window_0.ckpt"
# current_best_model_path = "current_best_model_79.ckpt"
saving_path = "pred_83.csv"

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    transforms.RandomRotation(30),
    transforms.RandomGrayscale(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(size=(128,128)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.ColorJitter(contrast=0.5, brightness=0.5),
    transforms.RandomErasing(),
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
tfm_unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
tfm_test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)


# Construct data loaders.
# train_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
# valid_loader = DataLoader(train_val_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


MyPseudoLabel_unlabeled_set = []
MyPseudoLabel_test_set = []

def semi_unlabeled_set(index):
    # print("ckpt 0: ",len(MyPseudoLabel_unlabeled_set))
    return MyPseudoLabel_unlabeled_set[index]

def semi_test_set(index):
    # print("ckpt 1: ",len(MyPseudoLabel_test_set))
    return MyPseudoLabel_test_set[index]

def get_pseudo_labels_0(dataset, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # Iterate over the dataset by batches.
    indice = []
    label = []
    idx = 0
    count = torch.zeros((11))
    # print("ckpt 1: ",len(MyPseudoLabel))
    for batch in dataloader:
        img, _= batch
        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)
        max_prob, arg = torch.max(probs,1)
        for i in range(len(probs)):
            if(max_prob[i] >= threshold):
                indice.append(idx)
                count[arg[i]]+=1
            label.append(int(arg[i]))
            idx+=1

        # Filter the data and construct a new dataset.
    # print("ckpt 2: ",len(MyPseudoLabel))
    # # Turn off the eval mode.
    model.train()
    print("Labeled class distribution: ",count)
    print(f"Labeled / Unlabeled: {int(sum(count)):05d}/{idx:05d}")

    global MyPseudoLabel_unlabeled_set, tfm_unlabeled_set
    MyPseudoLabel_unlabeled_set = label
    tfm_unlabeled_set.target_transform = semi_unlabeled_set
    subset = Subset(tfm_unlabeled_set, indice)
    return subset

def get_pseudo_labels_1(dataset, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    # Iterate over the dataset by batches.
    indice = []
    label = []
    idx = 0
    count = torch.zeros((11))
    # print("ckpt 1: ",len(MyPseudoLabel))
    for batch in dataloader:
        img, _= batch
        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)
        max_prob, arg = torch.max(probs,1)
        for i in range(len(probs)):
            if(max_prob[i] >= threshold):
                indice.append(idx)
                count[arg[i]]+=1
            label.append(int(arg[i]))
            idx+=1

    del dataloader
        # Filter the data and construct a new dataset.
    # print("ckpt 2: ",len(MyPseudoLabel))
    # # Turn off the eval mode.
    model.train()
    print("Labeled class distribution: ",count)
    print(f"Labeled / Unlabeled: {int(sum(count)):05d}/{idx:05d}")

    global MyPseudoLabel_test_set, tfm_test_set
    MyPseudoLabel_test_set = label
    tfm_test_set.target_transform = semi_test_set
    subset = Subset(tfm_test_set, indice)
    return subset

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize a model, and put it on the device specified.
model = Classifier_0().to(device)
# model = models.vgg16(num_classes=11,pretrained=False).to(device)

# # Load 之前的繼續跑
# model.load_state_dict(torch.load("model_window_0.ckpt"))

# # # Load current best model
# best_model = Classifier_0().to(device)
# best_model.load_state_dict(torch.load("model_window_0.ckpt"))

# # # Pseudo Labeling
# pseudo_set = get_pseudo_labels_0(unlabeled_set, best_model)
# unlabeled_loader = DataLoader(pseudo_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# test_pseudo_set = get_pseudo_labels_1(test_set, best_model)
# test_unlabeled_loader = DataLoader(test_pseudo_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# del best_model

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-5, momentum=0.9)
if Scheduler == "Plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=64, eps=1e-12 ,verbose=True)
elif Scheduler == "Cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=1e-8)

best_acc = 0.0
last_update = 0
updated_acc = False
unlabeled_loader = None
test_unlabeled_loader = None


print("Start Training.")

for epoch in range(n_epochs):
    # print("ckpt 0: ",len(MyPseudoLabel))

    if (best_acc >= 0.7) and updated_acc:
        do_semi = True
        updated_acc = False
    
    if do_semi:
        # best_model = models.vgg16(num_classes=11,pretrained=False).to(device)
        best_model = Classifier_0().to(device)
        best_model.load_state_dict(torch.load(model_path))
        pseudo_set = get_pseudo_labels_0(unlabeled_set, best_model)
        unlabeled_loader = DataLoader(pseudo_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        test_pseudo_set = get_pseudo_labels_1(test_set, best_model)
        test_unlabeled_loader = DataLoader(test_pseudo_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        del best_model
        do_semi = False

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()
    train_loss = []
    train_accs = []
    gradient_norm = []

    # Iterate the training set by batches.
    for batch in train_loader:

        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        gradient_norm.append(grad_norm)
        optimizer.step() # Update the parameters with computed gradients.

        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    if (unlabeled_loader is not None ) and (test_unlabeled_loader is not None) :
        for batch in unlabeled_loader:

            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            gradient_norm.append(grad_norm)
            optimizer.step() # Update the parameters with computed gradients.

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)
        
        for batch in test_unlabeled_loader:

            imgs, labels = batch
            logits = model(imgs.to(device))
            loss = criterion(logits, labels.to(device))
            optimizer.zero_grad()
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            gradient_norm.append(grad_norm)
            optimizer.step() # Update the parameters with computed gradients.

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
            train_loss.append(loss.item())
            train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in valid_loader:

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        with torch.no_grad():
          logits = model(imgs.to(device))
        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))
        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    gradient_norm = sum(gradient_norm) / len(gradient_norm) 

    # Print the information.
    print(f"({epoch + 1:03d}/{n_epochs:03d}), Train acc = {train_acc:.5f}, Valid acc = {valid_acc:.5f}, Gradient norm mean per epoch = {gradient_norm:.5f}")

    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), model_path)
        print(f"Best validation acc = {best_acc:.5f}")
        last_update = 0
        updated_acc = True
    else:
        last_update+=1

    if last_update > earlystop:
        print("Early stop at: ",epoch+1)
        break

    if Scheduler == "Plateau":
        scheduler.step(best_acc)
    elif Scheduler == "Cosine":
        scheduler.step()




# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
model = Classifier_0().to(device)
# model = models.vgg16(num_classes=11,pretrained=False).to(device)

model.load_state_dict(torch.load(model_path))
model.eval()
predictions = []
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())


# Save predictions into the file.
with open(saving_path, "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")

print("Saving: ",saving_path)
