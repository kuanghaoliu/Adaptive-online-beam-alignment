import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Model_3D(nn.Module):

    def __init__(self):
        super(Model_3D, self).__init__()

        # CNN preprocessing
        self.bn0 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64,
                               kernel_size=(1,3), stride=(1,3), padding=(0,1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=256,
                               kernel_size=(1,3), stride=(1,3), padding=(0,1))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256,
                               kernel_size=(1,3), stride=(1,3), padding=(0,1))
        self.bn3 = nn.BatchNorm2d(256)

        # LSTM layer for input learning
        self.lstm = nn.LSTM(input_size=256, hidden_size=256)

        # FC for output
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 64)


    def forward(self, x, pre_points = 9):

        # CNN preprocessing
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        P_dim_size = x.shape[3]
        x = nn.AvgPool2d(kernel_size=(1, P_dim_size))(x)
        x = torch.squeeze(x)
        x = x.permute(2, 0, 1)  # 交換順序
        # length * batch_size * dim
        y = x

        # the first point
        y = self.lstm(y)
        y = y[0]
        y1 = y
        y1 = self.drop(y1)
        y1 = self.fc(y1)
        y1 = torch.unsqueeze(y1, dim=0)
        temp = y1

        # other (pre_points - 1) points
        # NOTE: all the points in one period output the same result for conventional LSTM
        for num in range(pre_points - 1):  # 只是把它疊在一起，其實早就預測完了
            y1 = torch.cat([y1, temp], 0)

        result = y1
        
        return result
