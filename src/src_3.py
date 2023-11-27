import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F


class encoder_1(nn.Module):
    def __init__(self, l, dropout):
        super(encoder_1, self).__init__()

        self.pr_conv1=nn.Conv1d(1,16,6)
        self.pr_bn1=nn.BatchNorm1d(16)
        self.pr_conv2=nn.Conv1d(16,64,6)
        self.pr_bn2=nn.BatchNorm1d(64)
        self.pr_conv3=nn.Conv1d(64,16,5)
        self.pr_bn3=nn.BatchNorm1d(16)

        self.pr_linear1=nn.Linear(2272,256)
        self.pr_bn4=nn.BatchNorm1d(256)
        self.pr_linear2=nn.Linear(256,128)
        self.pr_bn5=nn.BatchNorm1d(128)
        self.pr_linear3=nn.Linear(128,64)

        self.maxpool=nn.MaxPool1d(2)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.pr_conv1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn1(x)
        x=self.dropout(x)

        x=self.pr_conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn2(x)
        x=self.dropout(x)

        x=self.pr_conv3(x)
        x=self.relu(x)
        x=self.pr_bn3(x)
        x=self.dropout(x)

        x=torch.flatten(x,start_dim=1)
        x=self.pr_linear1(x)
        x=self.relu(x)
        x=self.pr_bn4(x)
        x=self.dropout(x)

        x=self.pr_linear2(x)
        x=self.relu(x)
        x=self.pr_bn5(x)
        x=self.dropout(x)

        x=self.pr_linear3(x)
        return x

class encoder_2(nn.Module):
    def __init__(self, l, dropout):
        super(encoder_2, self).__init__()

        self.pr_conv1=nn.Conv1d(1,16,6)
        self.pr_bn1=nn.BatchNorm1d(16)
        self.pr_conv2=nn.Conv1d(16,64,6)
        self.pr_bn2=nn.BatchNorm1d(64)
        self.pr_conv3=nn.Conv1d(64,16,5)
        self.pr_bn3=nn.BatchNorm1d(16)

        self.pr_linear1=nn.Linear(2272,256)
        self.pr_bn4=nn.BatchNorm1d(256)
        self.pr_linear2=nn.Linear(256,128)
        self.pr_bn5=nn.BatchNorm1d(128)
        self.pr_linear3=nn.Linear(128,64)

        self.maxpool=nn.MaxPool1d(2)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.pr_conv1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn1(x)
        x=self.dropout(x)

        x=self.pr_conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn2(x)
        x=self.dropout(x)

        x=self.pr_conv3(x)
        x=self.relu(x)
        x=self.pr_bn3(x)
        x=self.dropout(x)

        x=torch.flatten(x,start_dim=1)
        x=self.pr_linear1(x)
        x=self.relu(x)
        x=self.pr_bn4(x)
        x=self.dropout(x)

        x=self.pr_linear2(x)
        x=self.relu(x)
        x=self.pr_bn5(x)
        x=self.dropout(x)

        x=self.pr_linear3(x)
        return x

class encoder_3(nn.Module):
    def __init__(self, l, dropout):
        super(encoder_3, self).__init__()

        self.pr_conv1=nn.Conv1d(1,16,6)
        self.pr_bn1=nn.BatchNorm1d(16)
        self.pr_conv2=nn.Conv1d(16,64,6)
        self.pr_bn2=nn.BatchNorm1d(64)
        self.pr_conv3=nn.Conv1d(64,16,5)
        self.pr_bn3=nn.BatchNorm1d(16)

        self.pr_linear1=nn.Linear(2272,256)
        self.pr_bn4=nn.BatchNorm1d(256)
        self.pr_linear2=nn.Linear(256,128)
        self.pr_bn5=nn.BatchNorm1d(128)
        self.pr_linear3=nn.Linear(128,64)

        self.maxpool=nn.MaxPool1d(2)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.pr_conv1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn1(x)
        x=self.dropout(x)

        x=self.pr_conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn2(x)
        x=self.dropout(x)

        x=self.pr_conv3(x)
        x=self.relu(x)
        x=self.pr_bn3(x)
        x=self.dropout(x)

        x=torch.flatten(x,start_dim=1)
        x=self.pr_linear1(x)
        x=self.relu(x)
        x=self.pr_bn4(x)
        x=self.dropout(x)

        x=self.pr_linear2(x)
        x=self.relu(x)
        x=self.pr_bn5(x)
        x=self.dropout(x)

        x=self.pr_linear3(x)
        return x

class encoder_4(nn.Module):
    def __init__(self, l, dropout):
        super(encoder_4, self).__init__()

        self.pr_conv1=nn.Conv1d(1,16,6)
        self.pr_bn1=nn.BatchNorm1d(16)
        self.pr_conv2=nn.Conv1d(16,64,6)
        self.pr_bn2=nn.BatchNorm1d(64)
        self.pr_conv3=nn.Conv1d(64,16,5)
        self.pr_bn3=nn.BatchNorm1d(16)

        self.pr_linear1=nn.Linear(2272,256)
        self.pr_bn4=nn.BatchNorm1d(256)
        self.pr_linear2=nn.Linear(256,128)
        self.pr_bn5=nn.BatchNorm1d(128)
        self.pr_linear3=nn.Linear(128,64)

        self.maxpool=nn.MaxPool1d(2)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.pr_conv1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn1(x)
        x=self.dropout(x)

        x=self.pr_conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn2(x)
        x=self.dropout(x)

        x=self.pr_conv3(x)
        x=self.relu(x)
        x=self.pr_bn3(x)
        x=self.dropout(x)

        x=torch.flatten(x,start_dim=1)
        x=self.pr_linear1(x)
        x=self.relu(x)
        x=self.pr_bn4(x)
        x=self.dropout(x)

        x=self.pr_linear2(x)
        x=self.relu(x)
        x=self.pr_bn5(x)
        x=self.dropout(x)

        x=self.pr_linear3(x)
        return x

class encoder_5(nn.Module):
    def __init__(self, l, dropout):
        super(encoder_5, self).__init__()

        self.pr_conv1=nn.Conv1d(1,16,6)
        self.pr_bn1=nn.BatchNorm1d(16)
        self.pr_conv2=nn.Conv1d(16,64,6)
        self.pr_bn2=nn.BatchNorm1d(64)
        self.pr_conv3=nn.Conv1d(64,16,5)
        self.pr_bn3=nn.BatchNorm1d(16)

        self.pr_linear1=nn.Linear(2272,256)
        self.pr_bn4=nn.BatchNorm1d(256)
        self.pr_linear2=nn.Linear(256,128)
        self.pr_bn5=nn.BatchNorm1d(128)
        self.pr_linear3=nn.Linear(128,64)

        self.maxpool=nn.MaxPool1d(2)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.pr_conv1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn1(x)
        x=self.dropout(x)

        x=self.pr_conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn2(x)
        x=self.dropout(x)

        x=self.pr_conv3(x)
        x=self.relu(x)
        x=self.pr_bn3(x)
        x=self.dropout(x)

        x=torch.flatten(x,start_dim=1)
        x=self.pr_linear1(x)
        x=self.relu(x)
        x=self.pr_bn4(x)
        x=self.dropout(x)

        x=self.pr_linear2(x)
        x=self.relu(x)
        x=self.pr_bn5(x)
        x=self.dropout(x)

        x=self.pr_linear3(x)
        return x

class encoder_6(nn.Module):
    def __init__(self, l, dropout):
        super(encoder_6, self).__init__()

        self.pr_conv1=nn.Conv1d(1,16,6)
        self.pr_bn1=nn.BatchNorm1d(16)
        self.pr_conv2=nn.Conv1d(16,64,6)
        self.pr_bn2=nn.BatchNorm1d(64)
        self.pr_conv3=nn.Conv1d(64,16,5)
        self.pr_bn3=nn.BatchNorm1d(16)

        self.pr_linear1=nn.Linear(2272,256)
        self.pr_bn4=nn.BatchNorm1d(256)
        self.pr_linear2=nn.Linear(256,128)
        self.pr_bn5=nn.BatchNorm1d(128)
        self.pr_linear3=nn.Linear(128,64)

        self.maxpool=nn.MaxPool1d(2)
        self.dropout=nn.Dropout(dropout)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.pr_conv1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn1(x)
        x=self.dropout(x)

        x=self.pr_conv2(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.pr_bn2(x)
        x=self.dropout(x)

        x=self.pr_conv3(x)
        x=self.relu(x)
        x=self.pr_bn3(x)
        x=self.dropout(x)

        x=torch.flatten(x,start_dim=1)
        x=self.pr_linear1(x)
        x=self.relu(x)
        x=self.pr_bn4(x)
        x=self.dropout(x)

        x=self.pr_linear2(x)
        x=self.relu(x)
        x=self.pr_bn5(x)
        x=self.dropout(x)

        x=self.pr_linear3(x)
        return x



class EN_1(nn.Module):
    def __init__(self, l, dropout):
        super(EN_1, self).__init__()
        self.encoder_1 = encoder_1(l, dropout)

    def forward(self, x):
        x = self.encoder_1(x)
        return x

class EN_2(nn.Module):
    def __init__(self, l, dropout):
        super(EN_2, self).__init__()
        self.encoder_2 = encoder_2(l, dropout)

    def forward(self, x):
        x = self.encoder_2(x)
        return x

class EN_3(nn.Module):
    def __init__(self, l, dropout):
        super(EN_3, self).__init__()
        self.encoder_3 = encoder_3(l, dropout)

    def forward(self, x):
        x = self.encoder_3(x)
        return x

class EN_4(nn.Module):
    def __init__(self, l, dropout):
        super(EN_4, self).__init__()
        self.encoder_4 = encoder_4(l, dropout)

    def forward(self, x):
        x = self.encoder_4(x)
        return x

class EN_5(nn.Module):
    def __init__(self, l, dropout):
        super(EN_5, self).__init__()
        self.encoder_5 = encoder_5(l, dropout)

    def forward(self, x):
        x = self.encoder_5(x)
        return x

class EN_6(nn.Module):
    def __init__(self, l, dropout):
        super(EN_6, self).__init__()
        self.encoder_6 = encoder_6(l, dropout)

    def forward(self, x):
        x = self.encoder_6(x)
        return x



class AE_1(nn.Module):
    def __init__(self, l, dropout):
        super(AE_1, self).__init__()
        self.encoder_1 = encoder_1(l, dropout,)

        self.de_linear1 = nn.Linear(64, 128)
        self.de_bn1 = nn.BatchNorm1d(128, affine=False)
        self.de_linear2 = nn.Linear(128, 256)
        self.de_bn2 = nn.BatchNorm1d(256, affine=False)
        self.de_linear3 = nn.Linear(256, 2272)
        self.de_bn3 = nn.BatchNorm1d(2272, affine=False)

        self.de_conv1 = nn.ConvTranspose1d(16, 64, 5, bias=False)
        self.de_bn4 = nn.BatchNorm1d(64, affine=False)
        self.de_conv2 = nn.ConvTranspose1d(64, 16, 6, bias=False)
        self.de_bn5 = nn.BatchNorm1d(16, affine=False)
        self.de_conv3 = nn.ConvTranspose1d(16, 1, 6, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        x = self.encoder_1(x)

        x = self.de_linear1(x)
        x = self.relu(x)
        x = self.de_bn1(x)
        x = self.dropout(x)

        x = self.de_linear2(x)
        x = self.relu(x)
        x = self.de_bn2(x)
        x = self.dropout(x)

        x = self.de_linear3(x)
        x = self.relu(x)
        x = self.de_bn3(x)
        x = self.dropout(x)

        x = x.view(b, 16, 142)
        x = self.de_conv1(x)
        x = self.relu(x)
        x = self.de_bn4(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv2(x)
        x = self.relu(x)
        x = self.de_bn5(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv3(x)
        return x

class AE_2(nn.Module):
    def __init__(self, l, dropout):
        super(AE_2, self).__init__()
        self.encoder_2 = encoder_2(l, dropout,)

        self.de_linear1 = nn.Linear(64, 128)
        self.de_bn1 = nn.BatchNorm1d(128, affine=False)
        self.de_linear2 = nn.Linear(128, 256)
        self.de_bn2 = nn.BatchNorm1d(256, affine=False)
        self.de_linear3 = nn.Linear(256, 2272)
        self.de_bn3 = nn.BatchNorm1d(2272, affine=False)

        self.de_conv1 = nn.ConvTranspose1d(16, 64, 5, bias=False)
        self.de_bn4 = nn.BatchNorm1d(64, affine=False)
        self.de_conv2 = nn.ConvTranspose1d(64, 16, 6, bias=False)
        self.de_bn5 = nn.BatchNorm1d(16, affine=False)
        self.de_conv3 = nn.ConvTranspose1d(16, 1, 6, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        x = self.encoder_2(x)

        x = self.de_linear1(x)
        x = self.relu(x)
        x = self.de_bn1(x)
        x = self.dropout(x)

        x = self.de_linear2(x)
        x = self.relu(x)
        x = self.de_bn2(x)
        x = self.dropout(x)

        x = self.de_linear3(x)
        x = self.relu(x)
        x = self.de_bn3(x)
        x = self.dropout(x)

        x = x.view(b, 16, 142)
        x = self.de_conv1(x)
        x = self.relu(x)
        x = self.de_bn4(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv2(x)
        x = self.relu(x)
        x = self.de_bn5(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv3(x)
        return x

class AE_3(nn.Module):
    def __init__(self, l, dropout):
        super(AE_3, self).__init__()
        self.encoder_3 = encoder_3(l, dropout,)

        self.de_linear1 = nn.Linear(64, 128)
        self.de_bn1 = nn.BatchNorm1d(128, affine=False)
        self.de_linear2 = nn.Linear(128, 256)
        self.de_bn2 = nn.BatchNorm1d(256, affine=False)
        self.de_linear3 = nn.Linear(256, 2272)
        self.de_bn3 = nn.BatchNorm1d(2272, affine=False)

        self.de_conv1 = nn.ConvTranspose1d(16, 64, 5, bias=False)
        self.de_bn4 = nn.BatchNorm1d(64, affine=False)
        self.de_conv2 = nn.ConvTranspose1d(64, 16, 6, bias=False)
        self.de_bn5 = nn.BatchNorm1d(16, affine=False)
        self.de_conv3 = nn.ConvTranspose1d(16, 1, 6, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        x = self.encoder_3(x)

        x = self.de_linear1(x)
        x = self.relu(x)
        x = self.de_bn1(x)
        x = self.dropout(x)

        x = self.de_linear2(x)
        x = self.relu(x)
        x = self.de_bn2(x)
        x = self.dropout(x)

        x = self.de_linear3(x)
        x = self.relu(x)
        x = self.de_bn3(x)
        x = self.dropout(x)

        x = x.view(b, 16, 142)
        x = self.de_conv1(x)
        x = self.relu(x)
        x = self.de_bn4(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv2(x)
        x = self.relu(x)
        x = self.de_bn5(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv3(x)
        return x

class AE_4(nn.Module):
    def __init__(self, l, dropout):
        super(AE_4, self).__init__()
        self.encoder_4 = encoder_4(l, dropout,)

        self.de_linear1 = nn.Linear(64, 128)
        self.de_bn1 = nn.BatchNorm1d(128, affine=False)
        self.de_linear2 = nn.Linear(128, 256)
        self.de_bn2 = nn.BatchNorm1d(256, affine=False)
        self.de_linear3 = nn.Linear(256, 2272)
        self.de_bn3 = nn.BatchNorm1d(2272, affine=False)

        self.de_conv1 = nn.ConvTranspose1d(16, 64, 5, bias=False)
        self.de_bn4 = nn.BatchNorm1d(64, affine=False)
        self.de_conv2 = nn.ConvTranspose1d(64, 16, 6, bias=False)
        self.de_bn5 = nn.BatchNorm1d(16, affine=False)
        self.de_conv3 = nn.ConvTranspose1d(16, 1, 6, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        x = self.encoder_4(x)

        x = self.de_linear1(x)
        x = self.relu(x)
        x = self.de_bn1(x)
        x = self.dropout(x)

        x = self.de_linear2(x)
        x = self.relu(x)
        x = self.de_bn2(x)
        x = self.dropout(x)

        x = self.de_linear3(x)
        x = self.relu(x)
        x = self.de_bn3(x)
        x = self.dropout(x)

        x = x.view(b, 16, 142)
        x = self.de_conv1(x)
        x = self.relu(x)
        x = self.de_bn4(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv2(x)
        x = self.relu(x)
        x = self.de_bn5(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv3(x)
        return x

class AE_5(nn.Module):
    def __init__(self, l, dropout):
        super(AE_5, self).__init__()
        self.encoder_5 = encoder_5(l, dropout,)

        self.de_linear1 = nn.Linear(64, 128)
        self.de_bn1 = nn.BatchNorm1d(128, affine=False)
        self.de_linear2 = nn.Linear(128, 256)
        self.de_bn2 = nn.BatchNorm1d(256, affine=False)
        self.de_linear3 = nn.Linear(256, 2272)
        self.de_bn3 = nn.BatchNorm1d(2272, affine=False)

        self.de_conv1 = nn.ConvTranspose1d(16, 64, 5, bias=False)
        self.de_bn4 = nn.BatchNorm1d(64, affine=False)
        self.de_conv2 = nn.ConvTranspose1d(64, 16, 6, bias=False)
        self.de_bn5 = nn.BatchNorm1d(16, affine=False)
        self.de_conv3 = nn.ConvTranspose1d(16, 1, 6, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        x = self.encoder_5(x)

        x = self.de_linear1(x)
        x = self.relu(x)
        x = self.de_bn1(x)
        x = self.dropout(x)

        x = self.de_linear2(x)
        x = self.relu(x)
        x = self.de_bn2(x)
        x = self.dropout(x)

        x = self.de_linear3(x)
        x = self.relu(x)
        x = self.de_bn3(x)
        x = self.dropout(x)

        x = x.view(b, 16, 142)
        x = self.de_conv1(x)
        x = self.relu(x)
        x = self.de_bn4(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv2(x)
        x = self.relu(x)
        x = self.de_bn5(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv3(x)
        return x

class AE_6(nn.Module):
    def __init__(self, l, dropout):
        super(AE_6, self).__init__()
        self.encoder_6 = encoder_6(l, dropout,)

        self.de_linear1 = nn.Linear(64, 128)
        self.de_bn1 = nn.BatchNorm1d(128, affine=False)
        self.de_linear2 = nn.Linear(128, 256)
        self.de_bn2 = nn.BatchNorm1d(256, affine=False)
        self.de_linear3 = nn.Linear(256, 2272)
        self.de_bn3 = nn.BatchNorm1d(2272, affine=False)

        self.de_conv1 = nn.ConvTranspose1d(16, 64, 5, bias=False)
        self.de_bn4 = nn.BatchNorm1d(64, affine=False)
        self.de_conv2 = nn.ConvTranspose1d(64, 16, 6, bias=False)
        self.de_bn5 = nn.BatchNorm1d(16, affine=False)
        self.de_conv3 = nn.ConvTranspose1d(16, 1, 6, bias=False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h = x.size()
        x = self.encoder_6(x)

        x = self.de_linear1(x)
        x = self.relu(x)
        x = self.de_bn1(x)
        x = self.dropout(x)

        x = self.de_linear2(x)
        x = self.relu(x)
        x = self.de_bn2(x)
        x = self.dropout(x)

        x = self.de_linear3(x)
        x = self.relu(x)
        x = self.de_bn3(x)
        x = self.dropout(x)

        x = x.view(b, 16, 142)
        x = self.de_conv1(x)
        x = self.relu(x)
        x = self.de_bn4(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv2(x)
        x = self.relu(x)
        x = self.de_bn5(x)
        x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2)

        x = self.de_conv3(x)
        return x