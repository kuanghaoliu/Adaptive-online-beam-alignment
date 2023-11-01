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

'---中心點展開範圍---'
def get_beam_range(center_beam, num_beams, total_beams):
    #center_beam = 中心beam, num_beams = 要左/右延伸的長度, total_beams
    # Compute the left and right edge indices of the beamforming range
    left_edge = max(center_beam - num_beams, 0)
    right_edge = min(center_beam + num_beams, total_beams - 1)
    # Adjust the left and right edges if the center beam is near the edge
    if center_beam - num_beams < 0:  # 如果太靠左邊
        right_edge = min(2*num_beams,total_beams - 1)
        left_edge = 0
    elif center_beam + num_beams >= total_beams-1:  # 如果太靠右邊
        left_edge = max(total_beams - 2*(num_beams+1)+1,0)
        right_edge = total_beams - 1
    return left_edge, right_edge
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
            left_boundary = max(total_beams - 2*(num_beams+1)+1,0)
    else:  # 如果是遞減的
        right_expand = 2
        left_expand = total_ -right_expand
        left_boundary = max(center_beam - left_expand, 0)
        right_boundary = min(center_beam + right_expand, total_beams - 1)
        if center_beam - left_expand < 0:  # 如果太靠左邊
            right_boundary = min(2*num_beams,total_beams - 1)
            left_boundary = 0
        elif center_beam + right_expand >= total_beams - 1:  # 如果太靠右邊
            right_boundary = total_beams - 1
            left_boundary = max(total_beams - 2*(num_beams+1)+1,0)


    return left_boundary, right_boundary
def get_beam_range_v3(label_increasing,center_beam, num_beams, total_beams):
    # center_beam = 中心beam, num_beams = 要左/右延伸的長度, total_beams
    # Compute the left and right edge indices of the beamforming range
    total_ = num_beams * 2
    temp = center_beam
    temp2 = center_beam
    index = [center_beam]
    if label_increasing:   # 如果是遞增的
        left_expand = 1  # 不留邊，不考慮noise太大的情況
        right_expand = total_-left_expand
        for e in range(left_expand) :
            temp =  torch.clamp(temp - 2,0, total_beams-1)  # 限制範圍，超過了就在邊邊
            index.append(temp)
        for e in range(right_expand) :
            temp2 = torch.clamp(temp2 + 2,0, total_beams-1)
            index.append(temp2)
        index.sort(reverse=False)

    else:  # 如果是遞減的
        right_expand = 1
        left_expand = total_ -right_expand
        for e in range(right_expand):
            temp = torch.clamp(temp + 2, 0, total_beams - 1)  # 限制範圍，超過了就在邊邊
            index.append(temp)
        for e in range(left_expand):
            temp2 = torch.clamp(temp2 - 2, 0, total_beams - 1)
            index.append(temp2)
        index.sort(reverse=True) # 按降序排列

    return torch.tensor(index)
'---找出對應的訓練label---'
def get_new_idx(batch_size,channels,labels,extend_beam_num,total,m):

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    if extend_beam_num == 32 :
        channel_few = torch.zeros((batch_size, 2, 10, 64), device=device)
    else:
        channel_few = torch.zeros((batch_size, 2, 10, 2 * extend_beam_num + 1), device=device)
    label_index = torch.zeros((batch_size, total), dtype=torch.int32, device=device)  # 64beam對應到31beam裡所產生的index
    Ext= torch.zeros((batch_size, 10,2), dtype=torch.int32, device=device)  # 延伸beam的範圍起點
    channel_train = channels[:, :, 0: total: (m + 1), :]  # 取出每個training的點
    label_train = labels[:, 0: total: (m + 1)]  # 取出每個training的點
    label_train = torch.cat([label_train[:, -1:], label_train[:, :-1]], dim=1)  # 往後移動一格，我要讓每次都拿到上一次的預測值
    label_train[:, 0] = label_train[:, 1]  # 第一次給正確的正~
    for s in range(batch_size):
        for i in range(10):
            Ext[s, i, 0], Ext[s, i, 1] = get_beam_range(label_train[s, i], extend_beam_num, 64)
            temp_start = Ext[s, i, 0]
            temp_end = Ext[s, i, 1] + 1
            temp = torch.arange(temp_start, temp_end)
            channel_few[s, :, i, :] = channel_train[s, :, i, temp_start:temp_end]  # 取出範圍內31beam的訊號
            for p in range(m + 1):  # test 時用不到
                idx = 10 * i + p
                if labels[s, idx] not in temp:  # 如果真實的beam超出我接收到的訊號範圍
                    label_index[s, idx] = (temp - labels[s, idx]).abs().argmin().item()  # 抓離他的值最近的index
                    global out_num
                    out_num += 1

    return Ext, channel_few,label_index
def get_new_idx_v2(batch_size,channels,labels,extend_beam_num,total,m):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if extend_beam_num == 32:
        channel_few = torch.zeros((batch_size, 2, 10, 64), device=device)
    else:
        channel_few = torch.zeros((batch_size, 2, 10, 2 * extend_beam_num + 1), device=device)
    label_index = torch.zeros((batch_size, total), dtype=torch.int32, device=device)  # 64beam對應到31beam裡所產生的index
    Ext= torch.zeros((batch_size, 10,2), dtype=torch.int32, device=device)  # 延伸beam的範圍起點
    channel_train = channels[:, :, 0: total: (m + 1), :]  # 取出每個training的點的訊號
    label_train = labels[:, 0: total: (m + 1)]  # 取出每個training的最佳label
    label_train = torch.cat([label_train[:, -1:], label_train[:, :-1]], dim=1)  # 往後移動一格，我要讓每次都拿到上一次的預測值
    label_train[:,0] = label_train[:,1] # 第一次給正確的值~
    label_increasing = torch.all(label_train[:, :-1] <= label_train[:, 1:], dim=1)  # 判斷我的label是不是遞增

    for s in range(batch_size):
        for i in range(10):
            Ext[s, i, 0], Ext[s, i, 1] = get_beam_range_v2(label_increasing[s], label_train[s, i], extend_beam_num, 64)
            temp_start = Ext[s, i, 0]
            temp_end = Ext[s, i, 1] + 1
            temp = torch.arange(temp_start, temp_end)  #　把選到的beam攤開
            channel_few[s, :, i, :] = channel_train[s, :, i, temp_start:temp_end]  # 64beam中選出31個範圍
            for p in range(1,m+1):  # 預測是從1~99、101~199
                idx = 10 * i + p
                if labels[s, idx] not in temp :  # 速度20以上，是有可能發生預測label並不在範圍裡的事情
                    label_index[s, idx] = (temp - labels[s, idx]).abs().argmin().item()  # 抓離他的值最近的index
                    global out_num
                    out_num += 1
                    #raise ValueError("真實的beam超出選擇範圍，該更換beam的選擇方法")
                    # print("真實的beam超出選擇範圍，使用最接近的beam")
                else:
                    label_index[s, idx] = torch.nonzero(torch.eq(temp, labels[s, idx]))[0].item()  # 抓出label對照區間，看index在哪

    return Ext, channel_few,label_index
def get_new_idx_v3(batch_size,channels,labels,extend_beam_num,total,m):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if extend_beam_num == 32:
        channel_few = torch.zeros((batch_size, 2, 10, 64), device=device)
    else:
        channel_few = torch.zeros((batch_size, 2, 10, 2 * extend_beam_num + 1), device=device)
    label_index = torch.zeros((batch_size, total), dtype=torch.int32, device=device)  # 64beam對應到31beam裡所產生的index
    index = torch.zeros((batch_size, 10, extend_beam_num*2+1),dtype=torch.int32, device=device)
    channel_train = channels[:, :, 0: total: (m + 1), :]  # 取出每個training的點的訊號
    label_train = labels[:, 0: total: (m + 1)]  # 取出每個training的最佳label
    label_train = torch.cat([label_train[:, -1:], label_train[:, :-1]], dim=1)  # 往後移動一格，我要讓每次都拿到上一次的預測值
    label_train[:,0] = label_train[:,1] # 假設第一個完美，我自己給值~
    label_increasing = torch.all(label_train[:, :-1] <= label_train[:, 1:], dim=1)  # 判斷我的label是不是遞增

    for s in range(batch_size):
        for i in range(10):
            index[s,i,:] = get_beam_range_v3(label_increasing[s], label_train[s, i], extend_beam_num, 64)
            channel_few[s, :, i, :] = channel_train[s, :, i, index[s,i,:].long()]  # 64beam中選出31個範圍
            for p in range(1,m+1):  # 預測是從1~99、101~199
                idx = 10 * i + p
                if labels[s, idx]  not in index[s,i,:]:
                    label_index[s, idx] = (index[s,i,:] - labels[s, idx]).abs().argmin().item()  # 抓離他的值最近的index
                    global out_num
                    out_num += 1
                else:
                    label_index[s, idx] = torch.nonzero(torch.eq(index[s,i,:], labels[s, idx]))[0].item()  # 抓出label對照區間，看index在哪

    return index.to(torch.int32), channel_few,label_index

'---計算正確率跟beformining gain---'
def calculate_acc_BL(out_tensor,labels,ext,P,N,m,BL,beam_power):

    # output
    out_tensor_np = out_tensor.cpu().detach().numpy()
    # optimal beam index label
    gt_labels = labels.cpu().detach().numpy()
    get_ext = ext.cpu().detach().numpy()
    gt_labels = gt_labels.transpose(1, 0)
    out_shape = gt_labels.shape
    # beam amplitude label
    beam_power = beam_power.cpu().detach().numpy()
    beam_power = beam_power.transpose(1, 0, 2)

    for i in range(out_shape[0]):  # 每個點
        for j in range(out_shape[1]):  # batch size
            if i % (m + 1) != 0:  # 注意這邊跳過training點
                loss_count = i // (m + 1)
                d_count = i % (m + 1) - 1
                train_ans = np.squeeze(out_tensor_np[d_count, loss_count, j, :])
                train_index = np.argmax(train_ans)  # 第j個batch size，第loss_count個訓練點、第d_count個點預測。此時求出來的是31beam重新編排的index
                temp = np.arange(get_ext[j,loss_count, 0], get_ext[j,loss_count, 1] + 1)  # 抓出原始範圍
                true_train_index = temp[train_index]  # 抓出估計的beam
                # counting accuracy
                if true_train_index == gt_labels[i, j] :
                    P  +=  1
                else:
                    N  +=  1
                # counting normalized beamforming gain
                if BL.ndim >= 2 :  # 判斷有BL的話才做計算
                    BL[loss_count, d_count] += (beam_power[i, j, true_train_index] / max(beam_power[i, j, :])) ** 2
                else:
                    continue  # 我希望跳到下一個j循環，使用break會直接跳下一個i


    # return P, N
def calculate_acc_BL_v2(out_tensor,labels,index,P,N,m,BL,beam_power):

    # output
    out_tensor_np = out_tensor.cpu().detach().numpy()
    # optimal beam index label
    gt_labels = labels.cpu().detach().numpy()
    get_index = index.cpu().detach().numpy()
    gt_labels = gt_labels.transpose(1, 0)
    out_shape = gt_labels.shape
    # beam amplitude label
    beam_power = beam_power.cpu().detach().numpy()
    beam_power = beam_power.transpose(1, 0, 2)
    train_index = np.argmax(out_tensor_np, axis=3)  # 先抓出預測出來的Y對應到的最大值index
    for i in range(out_shape[0]):  # 每個點
        for j in range(out_shape[1]):  # batch size
            if i % (m + 1) != 0:
                loss_count = i // (m + 1)
                d_count = i % (m + 1) - 1
                predict_beam = get_index[j, loss_count, train_index[d_count,loss_count,j]]  # index對應到真正的預測beam
                # counting accuracy
                if predict_beam == gt_labels[i, j] :
                    P  +=  1
                else:
                    N  +=  1
                # counting normalized beamforming gain
                if BL.ndim >= 2 :  # 判斷有BL的話才做計算
                    BL[loss_count, d_count] += (beam_power[i, j, predict_beam] / max(beam_power[i, j, :])) ** 2
                else:
                    continue  # 我希望跳到下一個j循環，使用break會直接跳下一個i

    # return P, N
def eval(info,model, loader,extend_beam_num, total, m):

    # input: trained model, test data loader, total (number of total points) and m (points to be predicted between two times of beam training)
    # output: accuracy, losses and normalized beamforming gain
    # reset dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loader.reset()
    # loss function
    criterion = nn.CrossEntropyLoss()
    # judge whether dataset is finished
    done = False
    # counting accurate prediction
    P = np.zeros((1), dtype=int)
    # counting inaccurate prediction
    N = np.zeros((1), dtype=int)


    # normalized beamforming gain, 10: beam training number, m: points to be predicted between two times of beam training
    BL = np.zeros((10, m))
    BL_sample = np.zeros((10, m, 2560))
    # running loss
    running_loss = 0
    # count batch number
    batch_num = 0
    batch_size = 32
    channel_total = []
    labels_total = []
    beam_power_total = []
    ext_total = []
    index_total = []
    channel_train_total = []

    # evaluate validation set
    while not done:
        # read files
        # channels: sequence of mmWave beam training received signal vectors
        # labels: sequence of optimal mmWave beam indices
        # beam power: sequence of mmWave beam training received signal power vectors

        channels, labels, beam_power, done, count = loader.next_batch()
        if labels.shape[0] == 0:
            break

        diff_ratio = (0.01) * torch.ones(batch_size, 99, 10)
        diff_ratio = diff_ratio.to(device)

        # 決定策略
        if change_Strategy == 1 :
            ext, channel_train, label_index = get_new_idx(batch_size, channels, labels, extend_beam_num, total, m)
            ext_total.append(ext.numpy())
        elif change_Strategy == 2 :
            ext, channel_train, label_index = get_new_idx_v2(batch_size, channels, labels, extend_beam_num, total, m)
            ext_total.append(ext.numpy())
        elif change_Strategy == 3 :
            index, channel_train, label_index = get_new_idx_v3(batch_size, channels, labels, extend_beam_num, total, m)
            index_total.append(index.numpy())

        # 存起來
        channel_total.append(channels.numpy())
        labels_total.append(labels.numpy())
        beam_power_total.append(beam_power.numpy())
        channel_train_total.append(channel_train.numpy())





        labels = labels.to(torch.int64)

        if count == True:
            batch_num += 1
            # predicted results
            # out_tensor.shape: pre_points * length * batch_size * num_of_beam
            out_tensor = model(channel_train, diff_ratio, pre_points = m)
            if change_Strategy == 1 or change_Strategy == 2:
                calculate_acc_BL(out_tensor, labels, ext, P, N, m, BL, beam_power)
            elif change_Strategy == 3:
                calculate_acc_BL_v2(out_tensor, labels, index, P, N, m, BL, beam_power)








    # average accuracy
    acur = P / (P + N)
    # average loss
    losses = running_loss / batch_num
    # average beam power loss
    BL = BL / batch_num / 32
    global out_num
    out_num = out_num / batch_num / 32
    # print results
    # print("Accuracy: %.3f" % (acur))
    # print("Loss: %.3f" % (losses))
    # print("Beam power loss:")
    # print(BL.T)

    data_channel = np.concatenate(channel_total, axis=0)
    data_labels = np.concatenate(labels_total, axis=0)
    data_beam_power = np.concatenate(beam_power_total, axis=0)
    data_channel_few = np.concatenate(channel_train_total, axis=0)

    if change_Strategy == 1 or change_Strategy == 2:
        data_ext = np.concatenate(ext_total, axis=0)
        # 儲存訓練資料
        mat_name = 'data_' + info + '.mat'
        sio.savemat(mat_name, {'channel': data_channel,
                               'label': data_labels,
                               'beam_power': data_beam_power,
                               'channel_few': data_channel_few,
                               'ext': data_ext})
    elif change_Strategy == 3:
        data_index = np.concatenate(index_total, axis=0)
        # 儲存訓練資料
        mat_name = 'few_data_' + info + '.mat'
        sio.savemat(mat_name, {'channel': data_channel,
                               'label': data_labels,
                               'beam_power': data_beam_power,
                               'channel_few': data_channel_few,
                               'index': data_index, })



    return acur, losses, BL
# main function for model training and evaluation
# output: accuracy, losses and normalized beamforming gain
def main():
    # first loop for different velocities
    for velocity in [5,10,15,20,25,30]:
        # save corresponding information
        # print("velocity:", velocity)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # print(device)

        t = 5   # training time
        batch_size = 32  # batch size
        total = 1000  # total points  時間槽有1000個
        m = 99  # the number of points to be predicted between two times of beam training
        # print('batch_size:%d' % (batch_size))
        extend_beam_num = 5
        beam_number = min(extend_beam_num * 2 + 1, 64)
        global change_Strategy  # 設定為全局變量來控制使用策略
        change_Strategy = 3
        global out_num
        out_num = 0
        version_name = '_ODE_2CNN_1LSTM_60epoch'
        info = 'few_v' + str(velocity) + '_a' + str(velocity * 0.2) + '_beam' + str(int(beam_number)) +'_Strategy' + str(change_Strategy) + version_name
        print("info", info)
        print("device:", device)
        print("velocity:", velocity)
        print("total timeslots:", total)
        print("number of ODE predict  in two training :", m)
        print("Strategy:", change_Strategy)
        print("beam number:", beam_number)
        print("simulation time:", t)
        print("batch size:", batch_size)

        path2 = r'C:\Users\user\Desktop\data\ODE_dataset(R1_test)_v' + str(velocity)
        # path2 = r'C:\Users\Aiden\Desktop\my_data\ODE_dataset(R1_test)_v' + str(velocity)
        eval_loader = test_Dataloader_3D(path=path2, batch_size=batch_size, device=device)

        # save results
        acur_eval = []
        loss_eval = []
        BL_eval = np.zeros((10, m, t))  # normalized beamforming gain

        # first loop for training runnings
        for tt in range(t):
            #print('Train %d times' % (tt))
            # load model
            model_name = info + '_' + str(tt) + '_MODEL.pkl'
            model = torch.load(model_name)
            model.to(device)

            # # print parameters
            # for name, param in model.named_parameters():
            #     print('Name:', name, 'Size:', param.size())

            model.eval()
            #print('the evaling set:')
            acur, losses, BL = eval(info,model, eval_loader, extend_beam_num, total, m)
            acur_eval.append(acur)
            loss_eval.append(losses)
            BL_eval[:, :, tt] = np.squeeze(BL)




            # save the results  為了方便在每個時間查看，但只要看最後一個t就好
            mat_name = 'test_' + info+'_'+str(tt) + '.mat'
            sio.savemat(mat_name, {'acur_eval': acur_eval,
                                   'loss_eval': loss_eval,
                                   'BL_eval': BL_eval,
                                   'out_num': out_num})


if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    days, remainder = divmod(int(execution_time), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("程式執行時間：{} 天 {} 小時 {} 分鐘".format(days, hours, minutes))