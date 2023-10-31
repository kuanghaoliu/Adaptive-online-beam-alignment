import torch
import numpy as np
from pmdarima import auto_arima
import timeit
from matplotlib import pyplot as plt
import scipy.io as sio
import os
from os import listdir
from os.path import isfile, join
import torch
import math
from collections import Counter

class test_Dataloader_3D():
    def __init__(self, path='', batch_size=32, device='cpu'):
        self.batch_size = batch_size
        self.device = device
        self.files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        for i, f in enumerate(self.files):
            if not f.split('.')[-1] == 'mat':
                del (self.files[i])
        self.reset()

    def reset(self):
        self.done = False
        self.unvisited_files = [f for f in self.files]

        # batch_size * 2 * length * num of beam
        self.buffer = np.zeros((0, 2, 4001, 64))

        # batch_size * length
        self.buffer_label = np.zeros((0, 4001))

        # batch_size * length * num of beam
        self.buffer_beam_power = np.zeros((0, 4001, 64))

    def load(self, file):
        data = sio.loadmat(file)

        channels = data['MM_data'] # beam training received signal
        labels = data['beam_label'] - 1 # optimal beam index label
        beam_power = data['beam_power'] # beam amplitude

        return channels, labels, beam_power

    def next_batch(self):
        done = False
        count = True

        # sequentially load data
        while self.buffer.shape[0] < self.batch_size:
            if len(self.unvisited_files) == 0:
                done = True
                count = False
                break
            channels, labels, beam_power = self.load(
                self.unvisited_files.pop(0))

            # load data into buffers
            self.buffer = np.concatenate((self.buffer, channels), axis=0)
            self.buffer_label = np.concatenate((self.buffer_label, labels), axis=0)
            self.buffer_beam_power = np.concatenate((self.buffer_beam_power, beam_power), axis=0)

        out_size = min(self.batch_size, self.buffer.shape[0])
        # get data from buffers
        batch_channels = self.buffer[0 : out_size, :, :, :]
        batch_labels = np.squeeze(self.buffer_label[0 : out_size, :])
        batch_beam_power = np.squeeze(self.buffer_beam_power[0:out_size, :, :])

        self.buffer = np.delete(self.buffer, np.s_[0 : out_size], 0)
        self.buffer_label = np.delete(self.buffer_label, np.s_[0 : out_size], 0)
        self.buffer_beam_power = np.delete(self.buffer_beam_power, np.s_[0:out_size], 0)

        # format transformation for reducing overhead
        batch_channels = np.float32(batch_channels)
        batch_labels = batch_labels.astype(int)
        batch_beam_power = np.float32(batch_beam_power)

        return torch.from_numpy(batch_channels).to(self.device), torch.from_numpy(batch_labels).to(
            self.device), torch.from_numpy(batch_beam_power).to(
            self.device), done, count
def calculate_acc_BL(predict_beam,labels,P,N,m,BL,beam_power):

    # optimal beam index label


    gt_labels =  labels.transpose(1, 0)
    out_shape = gt_labels.shape
    # beam amplitude label

    beam_power = beam_power.transpose(1, 0, 2)
    predict  = np.roll(predict_beam, shift= 1, axis=1)
    predict[:, 0, :] = predict[:, 1, :]   # 產生假設的點


    for i in range(out_shape[0]):  # 每個點
        for j in range(out_shape[1]):  # batch size
            if i % (m + 1) != 0:  # 注意這邊跳過training點
                loss_count = i // (m + 1)  # training
                d_count = i % (m + 1) - 1  # 切分點
                train_index = predict[j,loss_count, d_count]

                # counting accuracy
                if train_index == gt_labels[i, j] :
                    P  +=  1
                else:
                    N  +=  1
                # counting normalized beamforming gain
                BL[loss_count, d_count] += (beam_power[i, j, train_index] / max(beam_power[i, j, :])) ** 2


def main():

    for velocity in [5,10,15,20,25,30]:

        '---參數設定---'
        total  = 4000
        m = 99
        batch_size = 256
        sequence = 5
        version_name = '_ARIMA'
        beam_number = 64
        info = 'few_v' + str(velocity) + '_a' + str(velocity * 0.2) + version_name
        print("info", info)
        print("velocity:", velocity)
        print("total timeslots:", total)
        print("number of ODE predict  in two training:", m)
        print("beam number:", beam_number)
        print("batch size:", batch_size)



        '---程式開始---'
        # 讀取時間序列資料''
        path = r'C:\Users\Aiden\Desktop\my_data\time_01_SNR_104\final_4s\ODE_dataset(final_test)_v' + str(velocity)

        test_loader = test_Dataloader_3D(path=path, batch_size=batch_size)
        test_loader .reset()
        batch_num = 0
        BL = np.zeros((40, m))
        P = np.zeros((1), dtype=int)
        N = np.zeros((1), dtype=int)
        predict =  np.zeros((batch_size,40,m),dtype=int)
        done = False
        while not done:  # 有10個檔案，但每個檔案的batch會慢慢吐出來
            channels, labels, beam_power, done, count = test_loader.next_batch()
            if labels.shape[0] == 0:
                break
            channels = channels.numpy()
            labels = labels.numpy()
            beam_power = beam_power.numpy()

            training_num = 40
            signal = channels[:,0,:,:]+1j*channels[:,1,:,:]
            beam =np.argmax(np.abs(signal),axis = 2)  # 由Y取出beam



            if count == True:
                batch_num += 1
                print('batch_num:', batch_num)
                # 開始循環預測
                for b in range(batch_size):
                    for t in range(training_num):  # 每個seq代表一次訓練資料
                        data = beam[b,t*100+1+m-sequence+1:t*100+1+m+1]
                        model = auto_arima(data, seasonal=False, trace=False)
                        if model.order == (0,0,0):  # beam沒有變動
                            predict[b, t, :] = data[-1]
                        else:
                            temp = model.predict(n_periods = m)
                            predict[b,t,:] = np.clip(temp,0,63)   # index限制在此範圍
                calculate_acc_BL(predict, labels,  P, N, m, BL, beam_power)

        # average accuracy
        acur = P / (P + N)
        # average beam power loss
        BL = BL / batch_num / batch_size

        '---儲存資料---'
        mat_name = 'test_' + info + '.mat'
        sio.savemat(mat_name, {'acur_eval': acur,
                               'BL_eval': BL})


if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    days, remainder = divmod(int(execution_time), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("程式執行時間：{} 天 {} 小時 {} 分鐘".format(days, hours, minutes))