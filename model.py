import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()

        self.conv01 = nn.Conv2d(3, 64, 3)
        self.conv02 = nn.Conv2d(64, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv03 = nn.Conv2d(64, 128, 3)
        self.conv04 = nn.Conv2d(128, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv05 = nn.Conv2d(128, 256, 3)
        self.conv06 = nn.Conv2d(256, 256, 3)
        self.conv07 = nn.Conv2d(256, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv08 = nn.Conv2d(256, 512, 3)
        self.conv09 = nn.Conv2d(512, 512, 3)
        self.conv10 = nn.Conv2d(512, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(512, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.avepool1 = nn.AdaptiveAvgPool2d((7, 7))

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 3)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.relu(self.conv01(x))
        x = F.relu(self.conv02(x))
        x = self.pool1(x)

        x = F.relu(self.conv03(x))
        x = F.relu(self.conv04(x))
        x = self.pool2(x)

        x = F.relu(self.conv05(x))
        x = F.relu(self.conv06(x))
        x = F.relu(self.conv07(x))
        x = self.pool3(x)

        x = F.relu(self.conv08(x))
        x = F.relu(self.conv09(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.pool5(x)

        x = self.avepool1(x)

        # 行列をベクトルに変換
        x = x.view(-1, 512 * 7 * 7)
        
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.dropout1(x)
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

class MultiClassifier(nn.Module):
    def __init__(self):
        super(MultiClassifier, self).__init__()

        self.ConvLayer1 = nn.Sequential(
            # ref(H_out & W_out): https://pytorch.org/docs/stable/nn.html#conv2d
            nn.Conv2d(3, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            )

        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            )    

        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Dropout(0.2, inplace=True),
            )    

        self.Linear1 = nn.Linear(256 * 4 * 4, 2048)
        self.Linear2 = nn.Linear(2048, 1024)
        self.Linear3 = nn.Linear(1024, 512)
        self.Linear4 = nn.Linear(512, 3)


    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
#         print(x.size())
        x = x.view(-1, 256 * 4 * 4)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        x = self.Linear4(x)
        return x