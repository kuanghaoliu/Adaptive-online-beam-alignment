import torch.optim as optim
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
# dataloader for training
from train_dataloader_3D import train_Dataloader_3D
# dataloader for validating
from eval_dataloader_3D import eval_Dataloader_3D
# model for aperiodic training data
from model_lstm import Model_3D
import sys
import timeit

#挑選beam的範圍
def get_beam_range(center_beam, num_beams, total_beams):
    #center_beam = 中心beam, num_beams = 要左/右延伸的長度, total_beams
    # Compute the left and right edge indices of the beamforming range
    left_boundary = max(center_beam - num_beams, 0)
    right_boundary = min(center_beam + num_beams, total_beams - 1)
    # Adjust the left and right edges if the center beam is near the edge
    if center_beam - num_beams < 0:  # 如果太靠左邊
        right_boundary = min(2*num_beams,total_beams - 1)  # 會這樣設定是因為我目前這樣的展開方式無法讓總beam達到64beam，我為了跟原作比較，所以設beam的上限
        left_boundary = 0
    elif center_beam + num_beams >= total_beams-1:  # 如果太靠右邊
        left_boundary = max(total_beams - 2*(num_beams+1)+1,0)
        right_boundary = total_beams - 1

    return left_boundary, right_boundary

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




# 1.挑出最佳beam的延伸範圍 2.挑出training的點當作訓練點 3.選出特定範圍的beam，並且重新編號label
def get_new_idx(batch_size,channels,labels,extend_beam_num,total,m):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global change_Strategy
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
    label_train[:,0] = label_train[:,1]
    for s in range(batch_size):
        for i in range(10):
            Ext[s, i, 0], Ext[s, i, 1] = get_beam_range(label_train[s, i], extend_beam_num, 64)
            temp_start = Ext[s, i, 0]
            temp_end = Ext[s, i, 1] + 1
            temp = torch.arange(temp_start, temp_end)  #　把選到的beam攤開
            channel_few[s, :, i, :] = channel_train[s, :, i, temp_start:temp_end]  # 64beam中選出31個範圍
            for p in range(1,m+1):  # 預測是從1~99、101~199
                idx = 10 * i + p
                if labels[s, idx] not in temp :  # 速度20以上，是有可能發生預測label並不在範圍裡的事情
                    label_index[s, idx] = (temp - labels[s, idx]).abs().argmin().item()  # 抓離他的值最近的index
                    change_Strategy = True
                    print("偵測到UE速度過快")
                    break
                else:
                    label_index[s, idx] = torch.nonzero(torch.eq(temp, labels[s, idx]))[0].item()  # 抓出label對照區間，看index在哪
            if change_Strategy:  # 挑出迴圈
                print("跳出get_new_idx迴圈")
                break
        if change_Strategy:  # 挑出迴圈
            print("跳出get_new_idx迴圈")
            break
    if change_Strategy:  # 直接做V2版本
        print("調用get_new_idx_v2")
        Ext, channel_few, label_index = get_new_idx_v2(batch_size, channels, labels, extend_beam_num, total, m)

    return Ext, channel_few,label_index

def get_new_idx_v2(batch_size,channels,labels,extend_beam_num,total,m):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if extend_beam_num == 32:
        channel_few = torch.zeros((batch_size, 2, 10, 64), device=device)
    else:
        channel_few = torch.zeros((batch_size, 2, 10, 2 * extend_beam_num + 1), device=device)
    label_index = torch.zeros((batch_size, total), dtype=torch.int32, device=device)  # 64beam對應到31beam裡所產生的index
    Ext = torch.zeros((batch_size, 10,2), dtype=torch.int32, device=device)  # 延伸beam的範圍起點
    channel_train = channels[:, :, 0: total: (m + 1), :]  # 取出每個training的點的訊號
    label_train = labels[:, 0: total: (m + 1)]  # 取出每個training的最佳label
    label_train = torch.cat([label_train[:, -1:], label_train[:, :-1]], dim=1)  # 往後移動一格，我要讓每次都拿到上一次的預測值
    label_train[:,0] = label_train[:,1]
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
                    #raise ValueError("真實的beam超出選擇範圍，該更換beam的選擇方法")
                    # print("真實的beam超出選擇範圍，使用最接近的beam")
                    # global hahaha
                    # hahaha += 1
                    # print(hahaha)
                else:
                    label_index[s, idx] = torch.nonzero(torch.eq(temp, labels[s, idx]))[0].item()  # 抓出label對照區間，看index在哪

    return Ext, channel_few,label_index



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
    train_index = np.argmax(out_tensor_np, axis=3)  # 先抓出預測出來的Y對應到的最大值index
    for i in range(out_shape[0]):  # 每個點
        for j in range(out_shape[1]):  # batch size
            if i % (m + 1) != 0:
                loss_count = i // (m + 1)
                d_count = i % (m + 1) - 1
                #rain_ans = out_tensor_np[d_count, loss_count, j, :]
                #train_index = np.argmax(train_ans)  # 第j個batch size，第loss_count個訓練點、第d_count個點預測。此時求出來的是31beam重新編排的index
                temp = np.arange(get_ext[j,loss_count, 0], get_ext[j,loss_count, 1] + 1)  # 抓出原始範圍
                #true_train_index = temp[train_index]  # 抓出估計的beam
                true_train_index = temp[train_index[d_count, loss_count, j]]  # index對應到真正的預測beam
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


# 計算正確率
def calculate_acc_BL_eval(out_tensor,labels,ext,P,N,m,BL,beam_power):
    # count predicted instants between two times of beam training
    d_sery = np.linspace(np.float32(1) / m / 2, np.float32(1) - np.float32(1) / m / 2, num=m)
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
            if i % (m + 1) != 0:
                loss_count = i // (m + 1)
                d_count = i % (m + 1) - 1
                #rain_ans = out_tensor_np[d_count, loss_count, j, :]
                #train_index = np.argmax(train_ans)  # 第j個batch size，第loss_count個訓練點、第d_count個點預測。此時求出來的是31beam重新編排的index
                temp = np.arange(get_ext[j,loss_count, 0], get_ext[j,loss_count, 1] + 1)  # 抓出原始範圍
                #true_train_index = temp[train_index]  # 抓出估計的beam
                # select the nearest predicted instant
                t_d = np.abs(d_count * (1 / (m + 1)) + (1 / (m + 1)) - d_sery)
                min_location = np.argmin(t_d)
                # select the predicted result
                train_ans = np.squeeze(out_tensor_np[min_location, loss_count, j, :])
                # find the index with the maximum probability
                train_index = np.argmax(train_ans)  # 先抓出預測出來的Y對應到的最大值index
                true_train_index = temp[train_index]  # index對應到真正的預測beam

                t_d = np.abs(d_count * (1 / (m + 1)) + (1 / (m + 1)) - d_sery)
                min_location = np.argmin(t_d)


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

# model_evaluation
def eval(model, loader, extend_beam_num,total, m):
    # input: trained model, test data loader, total (number of total points) and m (points to be predicted between two times of beam training)
    # output: accuracy, losses and normalized beamforming gain
    # reset dataloader
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loader.reset()
    # global hahaha
    # hahaha = 0
    # loss function
    criterion = nn.CrossEntropyLoss()
    # judge whether dataset is finished
    done = False
    global P, N
    # counting accurate prediction
    P = np.zeros((1),dtype=int)
    # counting inaccurate prediction
    N = np.zeros((1),dtype=int)
    # normalized beamforming gain, 10: beam training number, m: points to be predicted between two times of beam training
    BL = np.zeros((10, m))
    # running loss
    running_loss = 0
    # count batch number
    batch_num = 0
    batch_size = 32
    to_remove = [i for i in range(0, 100, 10)]

    # evaluate validation set
    while not done:
        # read files
        # channels: sequence of mmWave beam training received signal vectors
        # labels: sequence of optimal mmWave beam indices
        # beam power: sequence of mmWave beam training received signal power vectors
        channels, labels, beam_power, done, count = loader.next_batch()
        if labels.shape[0] == 0:
            break

        # ratio = np.array([[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.1],[0.1]])
        # ratio = np.expand_dims(ratio, axis = 1)
        # ratio = np.repeat(ratio, 10, axis = 1)
        # ratio = np.expand_dims(ratio, axis = 0)
        # diff_ratio = np.repeat(ratio, batch_size, axis = 0)
        # diff_ratio = torch.from_numpy(diff_ratio)
        diff_ratio = (0.1) * torch.ones(batch_size, 9, 10)
        diff_ratio = diff_ratio.to(device)
        # print("evaluation:")
        # print(diff_ratio.shape)

         # 決定是否更換策略
        if change_Strategy:
            # 策略 2
            ext, channel_train, label_index = get_new_idx_v2(batch_size, channels, labels, extend_beam_num, total, m)
        else:
            # 策略 1
            ext, channel_train, label_index = get_new_idx(batch_size, channels, labels, extend_beam_num, total, m)

        labels = labels.to(torch.int64)
        label_index = label_index.to(torch.int64)

        if count == True:
            batch_num += 1
            # predicted results
            # out_tensor.shape: pre_points * length * batch_size * num_of_beam
            # out_tensor = model(channel_eval_few, diff_ratio, pre_points = m)
            out_tensor = model(channel_train, pre_points=m)
            # calculate loss function
            loss = 0
            # for all predictions after 10 beam trainings
            for loss_count in range(10):
             # for all predictions between two times of beam training
                for d_count in range(m):
            #         ##t_d = np.abs(d_count * (1 / (m + 1)) + (1 / (m + 1)) - d_sery)
            #         ##min_location = np.argmin(t_d)
            #         # output(batch size , class) 因為天線有64beam，所以不能隨意改output維度
                    loss += criterion(torch.squeeze(out_tensor[d_count, loss_count, :, :]),label_index[:, loss_count * (m + 1) + d_count + 1])

            calculate_acc_BL_eval(out_tensor, labels, ext,P,N, m, BL, beam_power)

            # label_index = torch.index_select(label_index, 1, torch.tensor([i for i in range(label_index.size(1)) if i not in to_remove]))
            # out_flat = out_tensor.transpose(0, 1).contiguous().view(-1, out_tensor.shape[3])
            # label_flat = label_index.contiguous().view(-1)
            # new_loss = criterion(out_flat, label_flat)
            # new_loss = new_loss * label_flat.size(0) / batch_size




            running_loss += loss.data.cpu()
            #running_loss += new_loss.data.cpu()
    # average accuracy
    acur = P / (P + N)
    # average loss
    losses = running_loss / batch_num
    # average beam power loss
    BL = BL / batch_num / 32
    # print results
    #print("Accuracy: %.3f" % (acur))
    # print("Loss: %.3f" % (losses))
    # print("Beam power loss:")
    # print(BL.T)

    return acur, losses, BL


# main function for model training and evaluation
# output: accuracy, losses and normalized beamforming gain
def main():
    # first loop for different velocities
    for velocity in [5,10,15,20,25,30]:
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        t = 1  # training time
        epoch = 60 # training epoch
        batch_size = 32 # batch size
        total = 100 # total points
        m = 9 # the number of points to be predicted between two times of beam training
        extend_beam_num = 5  # 是兩邊都拓展15，總共30+1個
        global change_Strategy  # 設定為全局變量來控制使用策略
        change_Strategy = True  # 都使用策略2
        version_name = '_LSTM_60epoch'
        # a = 0.2v m/s^2
        beam_number = min(extend_beam_num*2+1,64)
        info = 'few_v' + str(velocity) + '_a' + str(velocity * 0.2) +'_beam'+str(int(beam_number))+ version_name
        print("info", info)
        print("device:", device)
        print("velocity:", velocity)
        print("total timeslots:", total)
        print("number of ODE predict  in two training :", m)
        print("Strategy:",change_Strategy)
        print("beam number:", beam_number)
        print("simulation time:", t)
        print("epoch:", epoch)
        print("batch size:", batch_size)


        # dataset路徑
        path1 = r'C:\Users\Aiden\Desktop\my_data\time_01_SNR_104\ODE_dataset(v2)_v' + str(velocity) + '_training'
        path2 = r'C:\Users\Aiden\Desktop\my_data\time_01_SNR_104\ODE_dataset_v' + str(velocity) + '_evaluation'

        loader = train_Dataloader_3D(path=path1, batch_size=batch_size, device=device)
        eval_loader = eval_Dataloader_3D(path=path2, batch_size=batch_size, device=device)

        # loss function
        criterion = nn.CrossEntropyLoss()

        # save results
        acur_eval = []
        acur_train = []
        loss_eval = []
        BL_eval = np.zeros((10, m, epoch, t))  # normalized beamforming gain
        loss_train = []

        # first loop for training runnings
        for tt in range(t):  # 原作者做了6次平均，線會比較平滑一些
            # global hahaha  用來統計跳出範圍的次數，程式中有多處此類設定
            # hahaha = 0
            #print('Train %d times' % (tt))
            # learning rate
            lr = 0.00003
            # model initialization
            model = Model_3D()
            model.to(device)
            # Adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr, betas=(0.9, 0.999))
            # learning rate adaptive decay
            lr_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,verbose=False, threshold=0.0001,threshold_mode='rel', cooldown=0, min_lr=0.0000001,eps=1e-08)

            # second loop for training times
            for e in range(epoch):
                #print('Train %d epoch' % (e))
                # 成功次數
                P_ = np.zeros((1),dtype=int)
                # 失敗次數
                N_ = np.zeros((1),dtype=int)
                # reset the dataloader
                loader.reset()
                eval_loader.reset()
                # judge whether data loading is done
                done = False
                # running loss
                running_loss = 0
                # count batch number
                batch_num = 0
                while not done:
                    # read files
                    # channels: sequence of mmWave beam training received signal vectors
                    # labels: sequence of optimal mmWave beam indices
                    # beam power: sequence of mmWave beam training received signal power vectors
                    # ratio: time offset
                    channels, labels, beam_power, ratio, done, count = loader.next_batch()
                    if ratio.shape[0] == 0 :
                        break
                    # 決定是否更換策略
                    if change_Strategy:
                        # 策略 2
                        ext, channel_train, label_index = get_new_idx_v2(batch_size, channels, labels, extend_beam_num, total, m)
                    else:
                        # 策略 1
                        ext, channel_train, label_index = get_new_idx(batch_size, channels, labels, extend_beam_num, total,m)

                    # batch_size * 9 * 10
                    zero_ratio = torch.zeros((batch_size, 1, 10), device=device)

                    diff_ratio = ratio - torch.cat((zero_ratio, ratio[:, 0 : m - 1 : 1, :]), 1)  # 算出時間點i+1跟時間點i的時間差
                    # print("train:")
                    # print(diff_ratio.shape)
                    labels = labels.to(torch.int64)
                    label_index = label_index.to(torch.int64)
                    if count == True:
                        batch_num += 1
                        # predicted results
                        # out_tensor.shape: pre_points * length * batch_size * num_of_beam
                        out_tensor = model(channel_train, pre_points=m)
                        test_label = []
                        loss = 0
                        BL_ = np.array([0])  # 為了丟進去下方是子滿足條件
                        # count predicted instants between two times of beam training
                        # m = 9: d_sery=[0.1 : 0.9 : 0.1]
                        ##d_sery = np.linspace(np.float32(1) / m / 2, np.float32(1) - np.float32(1) / m / 2, num = m)
                        #for all predictions after 10 beam trainings
                        for loss_count in range(10):  #
                            # time offset between two times of beam training
                            for d_count in range(m):
                                ##t_d = np.abs(d_count * ( 1 / (m + 1)) + ( 1 / (m + 1)) - d_sery)
                                                  # min_location is the closest time stamp
                                ##min_location = np.argmin(t_d)
                                # calculate prediction loss
                                loss += criterion(torch.squeeze(out_tensor[d_count, loss_count, :, :]),label_index[:, loss_count * (m + 1) + d_count + 1])  # 每個training點都沒有算進去，很厲害的跳過了

                        # label_index = torch.index_select(label_index, 1, torch.tensor([i for i in range(label_index.size(1)) if i not in to_remove]))
                        # out_flat = out_tensor.transpose(0, 1).contiguous().view(-1, out_tensor.shape[3])
                        # label_flat = label_index.contiguous().view(-1)
                        # new_loss = criterion(out_flat,label_flat)
                        # new_loss = new_loss * label_flat.size(0)/batch_size

                        # gradient back propagation
                        loss.backward()
                        #new_loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        #running_loss += new_loss.item()

                        calculate_acc_BL(out_tensor,labels,ext,P_, N_,m,BL_,beam_power)  # P_和N_不用回傳，會跟著更改


                # print results
                losses = running_loss / batch_num  # 每32beam在上面一起算了，所以要除回來
                # print('[%d] loss: %.3f' %(e + 1, losses))

                # average accuracy
                acur_ = P_ / (P_ + N_)
                loss_train.append(losses)
                acur_train.append(acur_)
                # eval mode, where dropout is off
                model.eval()
                #print('the evaling set:')
                if 1:  # 決定要不要執行eval，如果很確定model可以成功的話，其實可以設為0
                    acur, losses, BL = eval(model, eval_loader,extend_beam_num, total, m)
                    acur_eval.append(acur)
                    loss_eval.append(losses)
                    BL_eval[:, :, e, tt] = np.squeeze(BL)


                # learning rate decay
                lr_decay.step(losses)
                # train mode, where dropout is on
                model.train()

                # save results into mat file
            mat_name = info + '.mat'
            sio.savemat(mat_name, {'acur_train':acur_train,'acur_eval': acur_eval,
                                   'loss_train': loss_train, 'loss_eval': loss_eval,
                                   'BL_eval': BL_eval})

            #save model
            model_name = info + '_' + str(tt) + '_MODEL.pkl'
            torch.save(model, model_name)

if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    days, remainder = divmod(int(execution_time), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("程式執行時間：{} 天 {} 小時 {} 分鐘".format(days, hours, minutes))