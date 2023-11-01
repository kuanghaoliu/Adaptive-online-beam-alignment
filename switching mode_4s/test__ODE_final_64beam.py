# Note: the trained models can be downloaded from the link https://cloud.tsinghua.edu.cn/d/e3b3793cb4ed4950be66/.

import torch.optim as optim
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
# dataloader for testing
from test_dataloader_3D import test_Dataloader_3D
import timeit
import sys
from sklearn.linear_model import LinearRegression
# model_evaluation



def calculate_threshold(BL):

    Beamformining_gain = np.array(BL)
    threshold = np.mean(Beamformining_gain[:,-1])


    # model = LinearRegression()
    # model.fit(X_train, y_train)

    return threshold
def get_beam_range_v2(label_increasing,center_beam, num_beams, total_beams):
    # center_beam = 中心beam, num_beams = 要左/右延伸的長度, total_beams
    # Compute the left and right edge indices of the beamforming range
    total_ = num_beams * 2
    if label_increasing:   # 如果是遞增的
        left_expand = 2
        right_expand = total_-left_expand
        left_boundary = max(center_beam - left_expand, 0)
        right_boundary = min(center_beam + right_expand, total_beams - 1)
        if center_beam - left_expand < 0:  # 如果太靠左邊
            right_boundary = min(2*num_beams,total_beams - 1)
            left_boundary = 0
        elif center_beam + right_expand >= total_beams - 1:  # 如果太靠右邊
            right_boundary = total_beams - 1
            left_boundary =  max(total_beams - 2*(num_beams+1)+1,0)
    else:  # 如果是遞減的
        right_expand = 2
        left_expand = total_ -right_expand
        left_boundary = max(center_beam - left_expand, 0)
        right_boundary = min(center_beam + right_expand, total_beams - 1)
        if center_beam - left_expand < 0:  # 如果太靠左邊
            right_boundary =  min(2*num_beams,total_beams - 1)
            left_boundary = 0
        elif center_beam + right_expand >= total_beams - 1:  # 如果太靠右邊
            right_boundary = total_beams - 1
            left_boundary =  max(total_beams - 2*(num_beams+1)+1,0)


    return left_boundary, right_boundary
def get_new_idx_v2(channel_train_31, label_total_31, seq_len, extend_beam_num):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    channel_few = torch.zeros((2, seq_len, 2 * extend_beam_num + 1), device=device)
    Ext = torch.zeros((seq_len, 2), dtype=torch.int32, device=device)  # 起始點跟結束點代表整個範圍
    label_train = torch.cat([label_total_31[-1:], label_total_31[:-1]])  # 往後移動一格，我要讓每次都拿到上一次的預測值
    label_train[0] = label_train[1]  # 第一次給正確的值~
    label_increasing = torch.all(label_train[:-1] <= label_train[1:])  # 判斷我的label是不是遞增

    for i in range(seq_len):
        Ext[i, 0], Ext[i, 1] = get_beam_range_v2(label_increasing, label_train[i], extend_beam_num, 64)  # 把31beam的範圍抓出來
        temp_start = Ext[i, 0]
        temp_end = Ext[i, 1] + 1
        channel_few[:, i, :] = channel_train_31[0, :, i, temp_start:temp_end]

    return Ext, channel_few
def calculate_acc_BL_few(out_tensor,labels,ext,m,beam_power,seq_len):


    P = np.zeros((seq_len,m))
    N = np.zeros((seq_len,m))
    acc = np.zeros((seq_len))
    BL = np.zeros((seq_len,m))
    # output
    out_tensor_np = out_tensor.cpu().detach().numpy()
    # optimal beam index label
    gt_labels = labels.cpu().detach().numpy()
    get_ext = ext.cpu().detach().numpy()
    # gt_labels = gt_labels.transpose(1, 0)
    out_shape = gt_labels.shape
    # beam amplitude label
    beam_power = beam_power.cpu().detach().numpy()
    # beam_power = beam_power.transpose(1, 0)
    for i in range(out_shape[0]):  # 每個點
        if i % (m + 1) != 0:  # 注意這邊跳過training點
            loss_count = i // (m + 1)
            d_count = i % (m + 1) - 1
            train_ans = np.squeeze(out_tensor_np[d_count, loss_count, 0, :])
            train_index = np.argmax(train_ans)  # 第j個batch size，第loss_count個訓練點、第d_count個點預測。此時求出來的是31beam重新編排的index
            temp = np.arange(get_ext[loss_count, 0], get_ext[loss_count, 1] + 1)  # 抓出原始範圍
            true_train_index = temp[train_index]  # 抓出估計的beam
            # counting accuracy
            if true_train_index == gt_labels[i] :
                P[loss_count, d_count]  =  1
            else:
                N[loss_count, d_count]  =  1
            # counting normalized beamforming gain
            BL[loss_count, d_count] = (beam_power[i, true_train_index] / max(beam_power[i, :])) ** 2

    for a in range(seq_len): # 算一下acc
        acc[a] = sum(P[a,:])/(sum(P[a,:])+sum(N[a,:]))

    return acc, BL
def calculate_acc_BL(out_tensor,labels,m,beam_power,seq_len):

    P = np.zeros((seq_len,m))
    N = np.zeros((seq_len,m))
    acc = np.zeros((seq_len))
    BL = np.zeros((seq_len,m))
    # output
    out_tensor_np = out_tensor.cpu().detach().numpy()
    # optimal beam index label
    gt_labels = labels.cpu().detach().numpy()
    # gt_labels = gt_labels.transpose(1, 0)
    out_shape = gt_labels.shape
    # beam amplitude label
    beam_power = beam_power.cpu().detach().numpy()
    # beam_power = beam_power.transpose(1, 0)
    for i in range(out_shape[0]):  # 每個點
        if i % (m + 1) != 0:  # 注意這邊跳過training點
            loss_count = i // (m + 1)
            d_count = i % (m + 1) - 1
            train_ans = np.squeeze(out_tensor_np[d_count, loss_count, 0, :])
            train_index = np.argmax(train_ans)  # 第j個batch size，第loss_count個訓練點、第d_count個點預測。此時求出來的是31beam重新編排的index
            # counting accuracy
            if train_index == gt_labels[i] :
                P[loss_count, d_count]  =  1
            else:
                N[loss_count, d_count]  =  1
            # counting normalized beamforming gain
            BL[loss_count, d_count] = (beam_power[i, train_index] / max(beam_power[i, :])) ** 2

    for a in range(seq_len):  # 算一下acc
        acc[a] = sum(P[a, :]) / (sum(P[a, :]) + sum(N[a, :]))

    return acc, BL
def eval(info,model,model2, loader,batch_size, seq_len, extend_beam_num, total, m,initial_switch):

    # input: trained model, test data loader, total (number of total points) and m (points to be predicted between two times of beam training)
    # output: accuracy, losses and normalized beamforming gain
    # reset dataloader
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loader.reset()
    # loss function
    #criterion = nn.CrossEntropyLoss()
    # judge whether dataset is finished
    done = False
   # count batch number
    batch_num = 0
    training = 0  # 我要計算總共使用了幾次training
    tracking = 0  # 我要計算總共使用了幾次tracking
    channel_store = []
    labels_store = []
    beam_power_store = []

    # evaluate validation set
    while not done:  # 有10個檔案，但每個檔案的batch會慢慢吐出來
        # read files
        # channels: sequence of mmWave beam training received signal vectors
        # labels: sequence of optimal mmWave beam indices
        # beam power: sequence of mmWave beam training received signal power vectors

        channels, labels, beam_power, done, count = loader.next_batch()
        # 儲存資料，做一次就好
        # channel_store.append(channels.numpy())
        # labels_store.append(labels.numpy())
        # beam_power_store.append(beam_power.numpy())
        if labels.shape[0] == 0:
            break

        diff_ratio = (0.01) * torch.ones(batch_size, 99, seq_len)  #
        diff_ratio = diff_ratio.to(device)
        labels = labels.to(torch.int64)
        channel_total = channels[:, :, 0: total: (m + 1), :]  # 每個時間點的64beam抓出來
        label_total = labels[0: total: (m + 1)]   # 把20每個訓練點label抓出來

        if count == True:
            # print(batch_num)
            batch_num += 1
            train_number = 0
            acc_batch = []
            BL_batch = []
            aver = []
            if batch_num == 1 :  # 先寫出來等方便存下來
                BL_average = np.zeros((channel_total.size(2), m))
                acc_average = np.zeros((channel_total.size(2)))

            # 先做第一次training預測，抓出4個預測點跟要預測的對應99個beam和99個power
            channel_train_initial = channel_total[:, :, train_number:train_number + seq_len, :]  # 抓出要訓練的4個點
            label_initial = labels[0:100 * (train_number + seq_len - 1) + 1 + m]  # 直接抓出預測的真實beam [1~399]
            beam_power_initial = beam_power[0:100 * (train_number + seq_len - 1) + 1 + m,:]


            if initial_switch :
                # 64beam 追蹤
                #pre_points * length * batch_size * num_of_beam
                out_tensor_64 = model(channel_train_initial, diff_ratio, timespans=seq_len, pre_points=m)  # 輸出最後一次切分的預測值
                acc, BL = calculate_acc_BL(out_tensor_64, label_initial, m, beam_power_initial, seq_len)
            else:
                #31beam 追蹤
                label_total_31 = label_total[train_number:train_number + seq_len]  # 抓出這次要用的所有最佳label點
                # ext, channel_train_31 = get_new_idx(channel_train_initial, label_total_31,seq_len, extend_beam_num)
                # 策略 2
                ext, channel_train_few = get_new_idx_v2(channel_train_initial, label_total_31,seq_len, extend_beam_num)

                # pre_points * length * batch_size * num_of_beam
                out_tensor_31 = model2(channel_train_few.unsqueeze(0), diff_ratio, timespans=seq_len, pre_points=m)  # 輸出最後一次切分的預測值
                acc, BL = calculate_acc_BL_few(out_tensor_31, label_initial, ext, m, beam_power_initial,seq_len)

            # 存下這次預測的acc和BL
            for i in range(seq_len):
                acc_batch.append(acc[i])
                BL_batch.append(BL[i,:])



            # 把剩下的時間序列做完
            for train_number in range(1,channel_total.size()[2]-seq_len+1):
                channel_train_initial = channel_total[:, :, train_number:train_number + seq_len, :]  # 抓出要使用的訓練點
                label_initial = labels[100 * train_number : 100 * (train_number+seq_len)]  # 抓出最後一次預測的真實beam [100~499]
                beam_power_initial = beam_power[100 * train_number : 100 * (train_number+seq_len), :]

                threshold = calculate_threshold(BL_batch)
                # threshold = calculate_threshold_v2(acc_batch)
                # if BL[-1,-1] > threshold :  # 拿預測完的最後一個判斷
                #
                if BL[-1, -1] > 2:  # 做31beam用的
                    trigger = True  # 使用beam training 31 beam
                else :
                    trigger = False # 使用beam training 64 beam

                if trigger:
                    # 使用beam training 31 beam
                    tracking += 1
                    label_total_31 = label_total[train_number:train_number + seq_len]  # 抓出這次要用的所有最佳label點
                    # ext, channel_train_31 = get_new_idx(channel_train_initial, label_total_31, seq_len, extend_beam_num)

                    # 策略 2
                    ext, channel_train_few = get_new_idx_v2(channel_train_initial, label_total_31, seq_len,extend_beam_num)
                    out_tensor_31 = model2(channel_train_few.unsqueeze(0), diff_ratio, timespans=seq_len,pre_points=m)  # timespans = 4 代表sequence，控制我一次進去的sequence長度，前三個是過去資料，第4個要當作我的輸出
                    acc, BL = calculate_acc_BL_few(out_tensor_31, label_initial, ext, m, beam_power_initial, seq_len)
                else:
                    # 使用beam training 64 beam
                    training += 1
                    out_tensor_64 = model(channel_train_initial, diff_ratio, timespans=seq_len, pre_points=m)  # 輸出最後一次切分的預測值
                    acc, BL = calculate_acc_BL(out_tensor_64, label_initial, m, beam_power_initial, seq_len)

                # 存下這次預測的acc和BL
                acc_batch.append(acc[-1])
                BL_batch.append(BL[-1,:])  # 我只要拿最後一個序列的預測值

            acc_average = acc_average + np.array(acc_batch)
            BL_average = BL_average + np.array(BL_batch)


    # average accuracy
    acc_average = acc_average/ batch_num
    BL_average = BL_average / batch_num
    training =  training / batch_num
    tracking = tracking / batch_num


    # average loss
    # losses = running_loss / batch_num
    # average beam power loss
    # BL = BL / batch_num / batch_size
    # print results
    # print("Accuracy: %.3f" % (acur))
    # print("Loss: %.3f" % (losses))
    # print("Beam power loss:")
    # print(BL.T)

    # mat_name = 'data_final.mat'
    # sio.savemat(mat_name, {'channel': channel_store,
    #                        'label': labels_store,
    #                        'beam_power': beam_power_store})


    return acc_average, BL_average, training ,tracking
def main():
    # first loop for different velocities
    for velocity in [30]:
        # save corresponding information
        # print("velocity:", velocity)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # print(device)

        t = 5   # training time
        batch_size = 1  # batch size
        total = 4000  # total points
        m = 99  # the number of points to be predicted between two times of beam training
        # print('batch_size:%d' % (batch_size))
        extend_beam_num = 5
        beam_number = min(extend_beam_num * 2 + 1, 64)
        seq_len = 10
        initial_switch = True # 起頭要用哪個方法  T是64beam F是31beam
        Strategy = 2 #  訓練跟測試必須使用同一種策略，因為我還沒有要討論低SNR情況，因此我都先使用策略2
        # 根據要產生的情況去選擇
        version_name = '64_switch_model_v'+ str(velocity) + '_a' + str(velocity * 0.2)+'_seq'+str(seq_len)
        # version_name = '11_switch_model_v' + str(velocity) + '_a' + str(velocity * 0.2) + '_seq' + str(seq_len)
        # version_name = 'switch_model_v' + str(velocity) + '_a' + str(velocity * 0.2) + '_seq' + str(seq_len)

        info = 'v' + str(velocity) + '_a' + str(velocity * 0.2) + '_beam64' + '_ODE_3CNN_1LSTM_60epoch'
        info_few = 'few_v' + str(velocity) + '_a' + str(velocity * 0.2) + '_beam' + str(beam_number)+'_Strategy'+str(Strategy)+ '_ODE_2CNN_1LSTM_60epoch'

        print("info", version_name)
        print('sequence:',seq_len)
        print("device:", device)
        print("velocity:", velocity)
        print("total timeslots:", total)
        print("number of ODE predict  in two training :", m)
        print("Strategy:", Strategy)
        print("initial_switch(T:64/ F:few) :",initial_switch)
        print("beam number:", beam_number)
        print("simulation time:", t)
        print("batch size:", batch_size)

        path = r'C:\Users\user\Desktop\data\final_4s\ODE_dataset(final_test)_v' + str(velocity)
        eval_loader = test_Dataloader_3D(path=path, batch_size=batch_size, device=device)

        # save results

        BL_eval = []  # normalized beamforming gain
        acur_eval = []
        train_eval = []
        track_eval = []
        # first loop for training runnings
        for tt in range(t):
            #print('Train %d times' % (tt))
            # load model
            model_name = info + '_' + str(tt) + '_MODEL.pkl'
            model_name_few = info_few + '_' + str(tt) + '_MODEL.pkl'
            model = torch.load(model_name,map_location=torch.device('cpu'))
            model2 = torch.load(model_name_few,map_location=torch.device('cpu'))
            model.to(device)
            model2.to(device)

            model.eval()
            model2.eval()
            #print('the evaling set:')

            acur_average, BL, training ,tracking= eval(info,model,model2, eval_loader,batch_size, seq_len, extend_beam_num, total, m,initial_switch)

            BL_eval.append(BL)
            acur_eval.append(acur_average)
            train_eval.append(training)
            track_eval.append(tracking)

        acur_eval = np.array(acur_eval)
        BL_eval = np.array(BL_eval)
        train_eval = np.array(train_eval)
        track_eval = np.array(track_eval)
        # save the results
        mat_name = 'final_test_' + version_name + '.mat'
        print("mat_name: ",mat_name)
        sio.savemat(mat_name, {'acur_eval': acur_eval,
                               'BL_eval': BL_eval,
                               'train_eval': train_eval,
                                'track_eval': track_eval})


if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    days, remainder = divmod(int(execution_time), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("程式執行時間：{} 天 {} 小時 {} 分鐘".format(days, hours, minutes))