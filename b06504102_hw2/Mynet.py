import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # 429 or 546
        # self.layer1 = nn.Linear(312, 1024)
        # self.layer1 = nn.Linear(220,1024)
        # self.layer1 = nn.Linear(429, 1024)
        # self.layer1 = nn.Linear(273,1024)
        self.layer1 = nn.Linear(546, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.layer4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        # self.layer5 = nn.Linear(1024, 1024)
        # self.bn5 = nn.BatchNorm1d(1024)
        # self.layer6 = nn.Linear(1024, 1024)
        # self.bn6 = nn.BatchNorm1d(1024)
        self.out = nn.Linear(1024, 39) 
        self.dropout = nn.Dropout(0.5)
        # self.act_fn = nn.Sigmoid()
        self.act_fn = nn.ReLU()

    def forward(self, x, target = None):
        # print(x.shape)
        x = self.layer1(x)
        x = self.act_fn(x)
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.act_fn(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.act_fn(x)
        x = self.bn3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.act_fn(x)
        x = self.bn4(x)
        x = self.dropout(x)
        

        # x = self.layer5(x)
        # x = self.act_fn(x)
        # x = self.bn5(x)
        # x = self.dropout(x)
        
        # x = self.layer6(x)
        # x = self.dropout(x)
        # x = self.bn6(x)
        # x = self.act_fn(x)

        x = self.out(x)
        
        return x