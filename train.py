import os

import numpy as np
from typing import NewType
from tqdm import tqdm
from torch.nn.modules import loss
from model import VGG, MultiClassifier
from data_input import EIDataset

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

def binary_acc(pred, label):
    
    total = label.size(0) * label.size(1)
    pred = torch.gt(pred, 0)
    label = torch.gt(label, 0)
    return (pred == label).sum().item(), total

def pred_acc(label, predicted):
    # ref: https://pytorch.org/docs/stable/torch.html#module-torch
    return torch.round(predicted).eq(label).sum().numpy()/len(label)

def visualize(epoch, train_, valid_, figname):
    fig = plt.figure()

    x = epoch
    y0 = train_
    y1 = valid_

    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

    plt.xlabel('epoch')
    plt.plot(x, y0, label='train_' + figname)
    plt.plot(x, y1, label='valid_' + figname)
    plt.legend()

    plt.savefig(figname + '.png')
    plt.close()

def split_layer(model):
    # featureモジュール
    params_to_update_1 = []
    # classifierモジュール(後半)
    params_to_update_2 = []
    # classifierモジュール(付け替えた層)
    params_to_update_3 = []

    # 学習させる層のパラメータ名を指定
    update_param_names_1 = ['features']
    update_param_names_2 = ['classifier.0.weight', 'classifier.0.bias',
                            'classifier.3.weight', 'classifier.3.bias']
    update_param_names_3 = ['classifier.6.weight', 'classifier.6.bias']

    # パラメータごとに各リストに格納
    for name, param in model.named_parameters():

        if update_param_names_1[0] in name:
            param.requires_grad = True
            params_to_update_1.append(param)
            print("params_to_update_1に格納：", name)
        
        elif name in update_param_names_2:
            param.requires_grad = True
            params_to_update_2.append(param)
            print("params_to_update_2に格納：", name)
        
        elif name in update_param_names_3:
            param.requires_grad = True
            params_to_update_3.append(param)
            print("params_to_update_3に格納：", name)

    return params_to_update_1 ,params_to_update_2, params_to_update_3

def train():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Use device : {}'.format(device))

    df = pd.read_csv('./img/labels.csv')

    train_df, valid_df = train_test_split(df, train_size=0.8)

    train_dataset = EIDataset(train_df, 'train')
    valid_dataset = EIDataset(valid_df, 'valid')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

    vgg = models.vgg11(pretrained=True)
    vgg.classifier[6] = nn.Linear(in_features=4096, out_features=3)
    vgg = vgg.to(device)
    feature_param, classifier_param, change_param = split_layer(vgg)

    optimizer = optim.SGD([
        {'params': feature_param, 'lr': 1e-4},
        {'params': classifier_param, 'lr': 5e-4},
        {'params': change_param, 'lr': 1e-3},
    ], lr=0.001, momentum=0.9, weight_decay=0.001)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 30
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

    for epoch in tqdm(range(1, epochs+1)):
        vgg.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train_total = 0
        train_running_acc = []
        for datas, labels in train_dataloader:
            optimizer.zero_grad()

            inputs = datas.to(device)
            labels = labels.float().to(device)

            outputs = vgg(inputs)

            train_loss = criterion(outputs, labels)
            correct, total = binary_acc(outputs, labels)
            acc_ = []
            for j, d in enumerate(outputs):
                acc = pred_acc(torch.Tensor.cpu(labels[j]), torch.Tensor.cpu(d))
                acc_.append(acc)

            total_train_loss += train_loss.item()
            total_train_correct += correct
            total_train_total += total
            train_running_acc.append(np.asarray(acc_).mean())

            train_loss.backward()
            optimizer.step()

        vgg.eval()
        smallest_loss = 10000
        total_valid_loss = 0
        total_valid_correct = 0
        total_valid_total = 0
        valid_running_acc = []
        with torch.no_grad():
            for datas, labels in valid_dataloader:
                inputs = datas.to(device)
                labels = labels.float().to(device)

                outputs = vgg(inputs)
                valid_loss = criterion(outputs, labels)
                correct, total = binary_acc(outputs, labels)
                acc_ = []
                for j, d in enumerate(outputs):
                    acc = pred_acc(torch.Tensor.cpu(labels[j]), torch.Tensor.cpu(d))
                    acc_.append(acc)

                total_valid_loss += valid_loss.item()
                total_valid_correct += correct
                total_valid_total += total
                valid_running_acc.append(np.asarray(acc_).mean())

        if total_valid_loss < smallest_loss:
            smallest_loss = total_valid_loss
            model_path = './best_model'
            torch.save(vgg.to('cpu').state_dict(), model_path)
            vgg.to(device)

        epoch_list.append(epoch)
        train_loss_list.append(total_train_loss)
        valid_loss_list.append(total_valid_loss)
        train_acc_list.append(total_train_correct/total_train_total)
        # train_acc_list.append(np.asarray(train_running_acc).mean())
        valid_acc_list.append(total_valid_correct/total_valid_total)
        # valid_acc_list.append(np.asarray(valid_running_acc).mean())

        visualize(epoch_list, train_loss_list, valid_loss_list, 'loss')
        visualize(epoch_list, train_acc_list, valid_acc_list, 'accuracy')

if __name__=='__main__':
    train()