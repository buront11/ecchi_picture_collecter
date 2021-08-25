from tqdm import tqdm
from torch.nn.modules import loss
from model import VGG
from data_input import EIDataset

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def binary_acc(pred, label):
    total = label.size(0) * label.size(1)
    pred = torch.gt(pred, 0.5)
    label = torch.gt(label, 0.5)
    return (pred == label).sum().item(), total

def visualize(epoch, train_, valid_, figname):
    fig = plt.figure()

    x = epoch
    y0 = train_
    y1 = valid_

    plt.xlabel('epoch')
    plt.plot(x, y0, label='train_loss')
    plt.plot(x, y1, label='valid_loss')
    plt.legend()

    plt.savefig(figname + '.png')
    plt.close()

def train():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EIDataset('./img/labels.csv')
    train_dataset, valid_dataset = train_test_split(dataset, train_size=0.9)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    vgg = VGG().to(device)
    optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 10
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    for epoch in tqdm(range(1, epochs)):
        vgg.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_total = 0
        for data, label in train_dataloader:
            optimizer.zero_grad()

            inputs = data.to(device)
            labels = label.float().to(device)

            outputs = vgg(inputs)

            train_loss = criterion(outputs, labels)
            correct, total = binary_acc(outputs, labels)

            total_train_loss += train_loss.item()
            total_train_correct += correct
            total_train_total += total

            train_loss.backward()
            optimizer.step()

        vgg.eval()
        total_valid_loss = 0
        total_valid_correct = 0
        total_valid_total = 0
        with torch.no_grad():
            for data, label in valid_dataloader:
                inputs = data.to(device)
                labels = label.float().to(device)

                outputs = vgg(inputs)
                valid_loss = criterion(outputs, labels)
                correct, total = binary_acc(outputs, labels)

                total_valid_loss += valid_loss.item()
                total_valid_correct += correct
                total_valid_total += total
        epoch_list.append(epoch)
        train_loss_list.append(total_train_loss)
        valid_loss_list.append(total_valid_loss)
        train_acc_list.append(total_train_correct/total_train_total)
        valid_acc_list.append(total_valid_correct/total_valid_total)

        visualize(epoch_list, train_loss_list, valid_loss_list, 'loss')
        visualize(epoch_list, train_acc_list, valid_acc_list, 'accuracy')

if __name__=='__main__':
    train()