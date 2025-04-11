import torch
import torch.nn as nn
from torch import nn
import numpy as np

def accuracy(y, y_hat):
    y_hat = y_hat.detach().numpy()
    correct = 0
    for i in range(y_hat.shape[0]):
        if round(y_hat[i]) == y[i]:
            correct +=1
    accuracy = correct/y_hat.shape[0]
    print(100*accuracy)
    return accuracy

class StackingModel(nn.Module):
    def __init__(self, args,):
        super().__init__()
        self.fc1 = nn.Linear(args.n_members*3, 50)
        if args.mode == 'stacking_naive' or args.mode == 'combine_naive':
            self.fc1 = nn.Linear(args.n_members, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
