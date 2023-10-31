import torch
import numpy as np
from pmdarima import auto_arima
import timeit
from matplotlib import pyplot as plt
import scipy.io as sio


def calculate_acc_BL(predict_beam,labels,UE_num, start,end,P,N,m,BL,beam_power,beam_number):

    # optimal beam index label
    gt_labels = labels.transpose(1, 0)
    out_shape = labels.shape
    # beam amplitude label
    beam_power_ = beam_power.transpose(1, 0, 2)
    true_train_ = np.full((UE_num,1000), np.nan)
    temp = np.zeros((beam_number),dtype = int)
    for i in range(out_shape[1]):  # 每個點
        for j in range(UE_num):  # batch size
            if i % (m + 1) != 0:  # 注意這邊跳過training點
                loss_count = i // (m + 1)  # training  0~9
                d_count = i % (m + 1) - 1  # 切分點
                train_index = predict_beam[j,loss_count, d_count]
                temp[:] = np.arange(start[j,loss_count], end[j,loss_count] )  # 抓出原始範圍
                true_train_index = temp[train_index]  # 抓出估計的beam
                true_train_[j,i] = true_train_index  # 最後99個沒用
    true_train_ = np.roll(true_train_, shift= 100, axis=1)  # 前99個不能算
    true_train_[:,:100 ] = true_train_ [:,100:200] # 還是要補上一個假的
    for i in range(out_shape[1]):  # 每個點
        for j in range(UE_num):  # batch size
            if i % (m + 1) != 0:  # 注意這邊跳過training點
                loss_count = i // (m + 1)  # training  0~9
                d_count = i % (m + 1) - 1  # 切分點
                # counting accuracy
                if int(true_train_[j,i]) == gt_labels[i, j] :
                    P  +=  1
                else:
                    N  +=  1
                # counting normalized beamforming gain
                BL[loss_count, d_count] += (beam_power_[i, j, int(true_train_[j,i])] / max(beam_power_[i, j, :])) ** 2

def main():

    for velocity in [30]:

        '---參數設定---'
        total  = 1000
        m = 99

        beam_number = 11
        strategy = 2
        version_name = '_ARIMA'
        # a = 0.2v m/s^2
        sequence = 99 # 控制數入的data數量

        info = 'few_v' + str(velocity) + '_a' + str(velocity * 0.2) + '_beam' + str(beam_number)+ '_data'+str(sequence)+ version_name
        print("info", info)
        print("velocity:", velocity)
        print("total timeslots:", total)
        print("number of ARIMA predict  in two training:", m)
        print("data:",sequence)
        print("beam number:", beam_number)
        print("Strategy:", strategy)




        '---讀取時間序列資料---'
        data_name = 'data_few_v'+str(velocity)+'_a' + str(velocity * 0.2) + '_beam' + str(beam_number)+'_Strategy'+str(strategy)+'_ODE_2CNN_1LSTM_60epoch.mat'
        data = sio.loadmat(data_name)
        channels = data['channel']  # 全部時間的64beam訊號
        labels = data['label']  # 真實最佳波束的Index
        beam_power = data['beam_power']  # 64
        channel_few = data['channel_few']  # 當下對齊週期的時間的11beam訊號
        ext = data['ext']  # 上面10個點的展開範圍，都是以前一個點為基準


        if velocity == 5 or velocity == 10:
            UE_num = 256

        elif velocity == 15:
            UE_num = 256
        elif velocity == 20 or velocity == 25 or velocity == 30:
            UE_num = 256

        training_num =  channel_few.shape[2] # 幾個訓練點
        BL = np.zeros((training_num, m))
        P = np.zeros((1), dtype=int) # 正確次數
        N = np.zeros((1), dtype=int) # 失敗次數
        predict =  np.zeros((UE_num,training_num,m),dtype=int)
        signal_few = np.zeros((sequence,beam_number), dtype = complex)  # 決定我要使用多少個歷史資訊
        training_data = np.zeros((sequence),dtype = int )

        start = ext[:, :, 0]
        end = ext[:, :, 1]+1  # python矩陣的設計，必須這樣才會取到11個beam
        signal = channels[:, 0, :, :] + 1j * channels[:, 1, :, :]  # 64beam訊號

        # 要改歷史資料長度，要改signal_few、training_data、 signal_few[b,:,:] = signal[b,t*100+1:t*100+1+m+1,start[b,t]:end[b,t]]
        # 開始循環預測
        for b in range(UE_num):  # 2560
            print("UE_num:",b)
        # for b in range(2):  # 2560
            for t in range(training_num):  # 10
                signal_few[:] = signal[b,t*100:t*100+m,start[b,t]:end[b,t]]
                training_data[:] = np.argmax(np.abs(signal_few[:]), axis=1) # 要放進去訓練的資料
                model = auto_arima(training_data, seasonal=False, trace=False)
                if model.order == (0,0,0):  # beam沒有變動
                    predict[b, t, :] = training_data[-1]
                else:
                    predict[b, t, :] = np.clip(model.predict(n_periods = m),0,beam_number-1)# index限制在此範圍
        calculate_acc_BL(predict, labels, UE_num, start, end, P, N, m, BL, beam_power,beam_number)

        # average accuracy
        acur = P / (P + N)
        # average beam power loss
        BL = BL / UE_num

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