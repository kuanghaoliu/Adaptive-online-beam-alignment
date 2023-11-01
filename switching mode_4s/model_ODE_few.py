import torch
import torch.nn as nn
import torch.nn.functional as F

# basic cell of ODE-LSTM with integrand function and integrating process between two cells  ODE架構及使用的類型
class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, solver_type):
        super(ODELSTMCell, self).__init__()
        # solver of ordinary differential equation
        self.solver_type = solver_type
        #print("solver_type:", self.solver_type)
        self.fixed_step_solver = solver_type.startswith("fixed_")  # 固定步長的ODE求解器

        # FC layer of neural ODE
        # fit the derivative
        self.f_node = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.input_size = input_size
        self.hidden_size = hidden_size

        # candidate integrand functions
        options = {
            "fixed_euler": self.euler,
            "fixed_heun": self.heun,
            "fixed_rk4": self.rk4,
        }
        if not solver_type in options.keys():  # 預設如果輸入不符合上面3種的ODE時，要跑出錯誤訊息
            raise ValueError("Unknown solver type '{:}'".format(solver_type))
        self.node = options[self.solver_type]

    def forward(self, new_h, ts):
        # feed the predicted results back
        # new_h is current_state
        # ts is integral duration
        new_h = self.solve_fixed(new_h, ts)
        return new_h

    def solve_fixed(self, x, ts):
        # integrate the predicted results
        ts = ts.view(-1, 1)  # 展平
        for i in range(3):  # 3 unfolds (refer to original ODE-LSTM paper)
            # integral step is ts * (1.0 / 3)
            x = self.node(x, ts * (1.0 / 3))
        return x

    # euler, heun and rk4 are different integration methods 三種ODE方法的執行架構
    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)
        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

# building model with ODE-LSTM cell  LSTM的機器學習架構(使用euler ODE)
class Model_3D(nn.Module):
    # model initialization
    def __init__(
        self,
        in_features=64,
        hidden_size=64,
        out_feature=11,
        return_sequences=True,
        solver_type="fixed_euler",
    ):
        super(Model_3D, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        # CNN preprocessing
        self.bn0 = nn.BatchNorm2d(2)  # number of features = 2
        self.conv1 = nn.Conv2d(in_channels=2, out_channels= 32,
                               kernel_size=(1, 3), stride=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels=64,
                               kernel_size=(1, 3), stride=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=256,
        #                        kernel_size=(1, 3), stride=(1, 3), padding=(0, 1))
        # self.bn3 = nn.BatchNorm2d(256)

        # define ODE-LSTM cell
        self.rnn_cell = ODELSTMCell(in_features, hidden_size, solver_type=solver_type)

        # LSTM layer for input learning
        self.lstm1 = nn.LSTMCell(input_size=64, hidden_size=64)

        # FC for output
        self.drop = nn.Dropout(0.3)  # 避免overfitting
        self.fc = nn.Linear(self.hidden_size, self.out_feature)

    # timespans: number of beam trainings
    # pre_points: number of predicted instants between two times of beam training
    def forward(self, x, time_ratio, timespans = 10, mask = None, pre_points = 9):  # 固定做10次training ，每次training間做9次預測
        device = x.device  # 還滿聰明的做法，看資料是放在哪裡，底下直接跟著設定
        batch_size = x.size(0)
        seq_len = timespans  # 10次training的數據一次輸入!?!??!?

        outputs = []
        last_output = torch.zeros((batch_size, self.out_feature), device=device)

        # CNN preprocessing
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = F.relu(x)

        P_dim_size = x.shape[3]
        x = nn.AvgPool2d(kernel_size=(1, P_dim_size))(x)  # pooling
        x = torch.squeeze(x,dim = 3)

        x = x.permute(0, 2, 1)  # 轉換維度位置順序成(seq,batch,hidden)
        time_ratio = time_ratio.permute(0, 2, 1)   # 代表兩個training中間切分的時間點

        # define variables of LSTM learning
        new_h = torch.zeros((batch_size, self.hidden_size), device=device)
        new_c = torch.zeros((batch_size, self.hidden_size), device=device)

        # ODE-LSTM learning
        for t in range(seq_len):  # 每個seq依序打入LSTM，每個seq代表一個training後的結果
            inputs = x[:, t]
            ratio1 = time_ratio[:, t]
            # LSTM learning
            new_h, new_c = self.lstm1(inputs, [new_h, new_c])

            # time offset
            # ts = [0.1, 0.1, 0.1, ...] if pre_points = 9
            ##ts = (1/(pre_points + 1)) * torch.ones(batch_size)
            ##ts = ts.to(device)

            # the first point
            # 1 * batch_size       第一個預測


            ts = ratio1[:, 0]
            new_h1 = self.rnn_cell.forward(new_h, ts)  # 丟入ODE-LSTMCell
            new_h1_temp = self.drop(new_h1)
            y1 = self.fc(new_h1_temp)
            y1 = torch.unsqueeze(y1, dim=0)

            # other (pre_points - 1) points  其他8個預測
            for num in range(pre_points-1):
                ts = ratio1[:, 1+num]
                new_h1 = self.rnn_cell.forward(new_h1, ts)
                new_h1_temp = self.drop(new_h1)
                y2 = self.fc(new_h1_temp)
                y2 = torch.unsqueeze(y2, dim=0)
                y1 = torch.cat([y1, y2], 0)

            # save the output
            current_output = y1  # 這裡是一個seq的所有預測點(9個)
            outputs.append(current_output)
            last_output = current_output

        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # return entire sequence  # 把第一維取出並疊在一起
        else:
            outputs = last_output  # only last item

        return outputs




