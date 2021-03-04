
#
#
#      0===========================================================0
#      |       TP6 PointNet for point cloud classification         |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 12/01/2021
#

import numpy as np
import random
import math
import os
import time
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F

# Import functions to read and write ply files
from ply import write_ply, read_ply



class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),      0],
                               [ math.sin(theta),  math.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.from_numpy(pointcloud)


def default_transforms():
    return transforms.Compose([RandomRotation_z(),RandomNoise(),ShufflePoints(),ToTensor()])


class PointCloudData(Dataset):
    def __init__(self, root_dir, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir+"/"+dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir+"/"+category+"/"+folder
            for file in os.listdir(new_dir):
                if file.endswith('.ply'):
                    sample = {}
                    sample['ply_path'] = new_dir+"/"+file
                    sample['category'] = category
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ply_path = self.files[idx]['ply_path']
        category = self.files[idx]['category']
        data = read_ply(ply_path)
        pointcloud = self.transforms(np.vstack((data['x'], data['y'], data['z'])).T)
        return {'pointcloud': pointcloud, 'category': self.classes[category]}


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=torch.relu, dropout=0):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.nonlin = nonlinearity
        self.do = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return self.bn(self.do(self.nonlin(self.fc(x))))


class Conv1d_block(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=1, nonlinearity=torch.relu, dropout=0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size)
        self.nonlin = nonlinearity
        self.do = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        return self.bn(self.do(self.nonlin(self.conv1(x))))


class PointMLP(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.fc1 = FC_block(3072, 512)
        self.fc2 = FC_block(512, 256, dropout=0.3)
        self.fc3 = FC_block(256, classes)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h = x.reshape(-1, 3072)
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        o = self.activation(h)
        return o


class PointNetBasic(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.cb1 = Conv1d_block(3, 64)
        self.cb2 = Conv1d_block(64, 64)
        self.cb3 = Conv1d_block(64, 64)
        self.cb4 = Conv1d_block(64, 128)
        self.cb5 = Conv1d_block(128, 1024)
        self.mp = nn.MaxPool1d(1024)
        self.fc1 = FC_block(1024, 512)
        self.fc2 = FC_block(512, 256, dropout=0.3)
        self.fc3 = nn.Linear(256, classes)
        self.activation = nn.LogSoftmax(dim=1)


    def forward(self, x):
        h = self.cb1(x)
        h = self.cb2(h)
        h = self.cb3(h)
        h = self.cb4(h)
        h = self.cb5(h)
        h = self.mp(h).squeeze()
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        o = self.activation(h)
        return o

        
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.cb1 = Conv1d_block(3, 64)
        self.cb2 = Conv1d_block(64, 128)
        self.cb3 = Conv1d_block(128, 1024)
        self.mp = nn.MaxPool1d(1024)
        self.fc1 = FC_block(1024, 512)
        self.fc2 = FC_block(512, 256)
        self.fc3 = nn.Linear(256, k * k)


    def forward(self, x):
        h = self.cb1(x)
        h = self.cb2(h)
        h = self.cb3(h)
        h = self.mp(h).squeeze()
        h = self.fc1(h)
        h = self.fc2(h)
        o = self.fc3(h)
        return o.reshape(-1, 3, 3)

class PointNetFull(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.tn1 = Tnet(3)
        self.cb1 = Conv1d_block(3, 64)
        self.cb2 = Conv1d_block(64, 64)
        # self.tn2 = Tnet(64)
        self.cb3 = Conv1d_block(64, 64)
        self.cb4 = Conv1d_block(64, 128)
        self.cb5 = Conv1d_block(128, 1024)
        self.mp = nn.MaxPool1d(1024)
        self.fc1 = FC_block(1024, 512)
        self.fc2 = FC_block(512, 256, dropout=0.3)
        self.fc3 = nn.Linear(256, classes)
        self.activation = nn.LogSoftmax(dim=1)


    def forward(self, x):
        m3x3 = self.tn1(x)
        h = torch.bmm(m3x3, x)
        h = self.cb1(h)
        h = self.cb2(h)
        # m64 = self.tn2()
        h = self.cb3(h)
        h = self.cb4(h)
        h = self.cb5(h)
        h = self.mp(h).squeeze()
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.fc3(h)
        o = self.activation(h)
        return o, m3x3




def basic_loss(outputs, labels):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    return criterion(outputs, labels)

def pointnet_full_loss(outputs, labels, m3x3, alpha = 0.001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)) / float(bs)




def train(model, device, train_loader, test_loader=None, epochs=250):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss=0
    for epoch in range(epochs): 
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            # outputs = model(inputs.transpose(1,2))
            outputs, m3x3 = model(inputs.transpose(1,2))
            # loss = basic_loss(outputs, labels)
            loss = pointnet_full_loss(outputs, labels, m3x3)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        if test_loader:
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    # outputs = model(inputs.transpose(1,2))
                    outputs, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Epoch: %d, Loss: %.3f, Test accuracy: %.1f %%' %(epoch+1, loss, val_acc))

        scheduler.step()


if __name__ == '__main__':
    
    t0 = time.time()
    
    train_ds = PointCloudData("../data/ModelNet40_PLY")
    test_ds = PointCloudData("../data/ModelNet40_PLY", folder='test')

    inv_classes = {i: cat for cat, i in train_ds.classes.items()}
    print("Classes: ", inv_classes)
    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.classes))
    print('Sample pointcloud shape: ', train_ds[0]['pointcloud'].size())

    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=32)

    # model = PointMLP()
    # model = PointNetBasic()
    model = PointNetFull()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print("Number of parameters in the Neural Networks: ", sum([np.prod(p.size()) for p in model_parameters]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("Device: ", device)
    model.to(device)

    train(model, device, train_loader, test_loader, epochs = 250)
    
    print("Total time for training : ", time.time()-t0)


