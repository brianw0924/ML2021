import torch
import torch.nn as nn
import torch.nn.functional as F


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
#         self.in_features = in_features
#         self.out_features = out_features
#         self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, wf, labels):
        '''
        input shape (N, in_features)
        '''
#         assert len(x) == len(labels)
#         assert torch.min(labels) >= 0
#         assert torch.max(labels) < self.out_features
        
#         for W in self.fc.parameters():
#             W = F.normalize(W, p=2, dim=1)

#         x = F.normalize(x, p=2, dim=1)

#         wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        # self.layer4 = nn.Linear(512, 512)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.layer5 = nn.Linear(512, 512)
        # self.bn5 = nn.BatchNorm1d(512)
        # self.layer6 = nn.Linear(512, 512)
        # self.bn6 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, 39, bias=False) 
        # self.out = nn.Linear(512, 39) 
        self.dropout = nn.Dropout(0.2)
        self.act_fn = nn.Sigmoid()

    def forward(self, x, target = None):
        # print(x.shape)
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.dropout(x)
        x = self.bn2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.dropout(x)
        x = self.bn3(x)
        x = self.act_fn(x)

        # x = self.layer4(x)
        # x = self.dropout(x)
        # x = self.bn4(x)
        # x = self.act_fn(x)

        # x = self.layer5(x)
        # x = self.dropout(x)
        # x = self.bn5(x)
        # x = self.act_fn(x)
        
#         x = self.layer6(x)
#         x = self.dropout(x)
#         x = self.bn6(x)
#         x = self.act_fn(x)

        self.out.weight.data = F.normalize(self.out.weight.data, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1) # why?

        x = self.out(x)
        
        return x