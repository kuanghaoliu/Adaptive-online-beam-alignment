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

# model_evaluation
def eval(model, loader, total, m):
    # input: trained model, test data loader, total (number of total points) and m (points to be predicted between two times of beam training)
    # output: accuracy, losses and normalized beamforming gain
    # reset dataloader
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    loader.reset()
    # loss function
    criterion = nn.CrossEntropyLoss()
    # judge whether dataset is finished
    done = False
    # counting accurate prediction
    P = 0
    # counting inaccurate prediction
    N = 0
    # normalized beamforming gain, 10: beam training number, m: points to be predicted between two times of beam training
    BL = np.zeros((10, m))
    BL_sample = np.zeros((10, m, 2560))
    # running loss
    running_loss = 0
    # count batch number
    batch_num = 0
    batch_size = 32

    # evaluate validation set
    while not done:
        # read files
        # channels: sequence of mmWave beam training received signal vectors
        # labels: sequence of optimal mmWave beam indices
        # beam power: sequence of mmWave beam training received signal power vectors
        channels, labels, beam_power, done, count = loader.next_batch()
        if labels.shape[0] == 0:
            break
        true_beam = np.array(labels)
        predict_beam = np.array(labels)
        diff_ratio = (0.01) * torch.ones(batch_size, 99, 10)
        diff_ratio = diff_ratio.to(device)

        labels = labels.to(torch.int64)
        # select data for beam training
        channel_train = channels[:, :, 0 : total : (m + 1), :]  # 0~1000 每99個挑一次起來，直接找beam training 的點

        if count == True:
            batch_num += 1  # 代表我會吐出幾次batch
            # predicted results
            # out_tensor.shape: pre_points * length * batch_size * num_of_beam
            out_tensor = model(channel_train, diff_ratio, pre_points = m)
            # calculate loss function
            loss = 0
            # for all predictions after 10 beam trainings
            for loss_count in range(10):
                # for all predictions between two times of beam training
                for d_count in range(m):
                    loss += criterion(torch.squeeze(out_tensor[d_count, loss_count, :, :]),
                                      labels[:, loss_count * (m + 1) + d_count + 1])
            # output
            out_tensor_np = out_tensor.cpu().detach().numpy()
            # optimal beam index label
            gt_labels = labels.cpu().detach().numpy()
            gt_labels = np.float32(gt_labels)
            gt_labels = gt_labels.transpose(1, 0)
            # beam amplitude label
            beam_power = beam_power.cpu().detach().numpy()
            beam_power = beam_power.transpose(1, 0, 2)

            out_shape = gt_labels.shape
            for i in range(out_shape[0]):
                # i = (10 beam trainings + 10 beam trainings X m predictions, in time order)
                for j in range(out_shape[1]):
                    # j from 0 to batch_size - 1
                    # the instants of beam training will not be predicted
                    if i % (m + 1) != 0:
                        # number of beam trainings
                        loss_count = i // (m + 1)
                        # time offset between two times of beam training
                        d_count = i % (m + 1) - 1
                        # select the nearest predicted instant (indeed, the instant is perfectly matched for ODE)
                        # select the predicted result
                        train_ans = np.squeeze(out_tensor_np[d_count, loss_count, j, :])
                        # find the index with the maximum probability
                        train_index = np.argmax(train_ans)
                        predict_beam[j,i] = train_index
                        # counting accurate and inaccurate prediction
                        if train_index == gt_labels[i, j]:
                            P = P + 1
                        else:
                            N = N + 1
                        # counting normalized beamforming gain
                        BL[loss_count, d_count] = BL[loss_count, d_count] + (beam_power[i, j, train_index] / max(
                            beam_power[i, j, :])) ** 2

            running_loss += loss.data.cpu()
    # average accuracy
    acur = float(P) / (P + N)
    # average loss
    losses = running_loss / batch_num
    # average beam power loss
    BL = BL / batch_num / 32
    # print results
    # print("Accuracy: %.3f" % (acur))
    # print("Loss: %.3f" % (losses))
    # print("Beam power loss:")
    # print(BL.T)

    return acur, losses, BL , predict_beam, true_beam


# main function for model training and evaluation
# output: accuracy, losses and normalized beamforming gain
def main():
    # first loop for different velocities
    for velocity in [5,10,15,20,25,30]:
        # save corresponding information
        # print("velocity:", velocity)
        version_name = '_ODE_3CNN_1LSTM_60epoch'
        # a = 0.2v m/s^2
        info ='v' + str(velocity) + '_a' + str(velocity * 0.2) + '_beam64' + version_name
        # print(info)
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # print(device)

        t = 5   # training time
        batch_size = 32  # batch size
        total = 1000  # total points  時間槽有1000個
        m = 99  # the number of points to be predicted between two times of beam training
        # print('batch_size:%d' % (batch_size))

        path2 = r'C:\Users\user\Desktop\data\ODE_dataset(R1_test)_v' + str(velocity)

        eval_loader = test_Dataloader_3D(path=path2, batch_size=batch_size, device=device)

        # save results
        acur_eval = []
        loss_eval = []
        BL_eval = np.zeros((10, m, t))  # normalized beamforming gain

        # first loop for training runnings
        for tt in range(t):
            print('Train %d times' % (tt))
            # load model
            model_name = info + '_' + str(tt) + '_MODEL.pkl'
            model = torch.load(model_name ,map_location=torch.device('cpu'))
            #model = torch.load(model_name)  如果模型原本是GPU但樣在CPU環境執行，記得用上面的code
            model.to(device)

            # print parameters
            # for name, param in model.named_parameters():
            #     print('Name:', name, 'Size:', param.size())


            model.eval()
            # print('the evaling set:')

            acur, losses, BL, predict_beam, true_beam = eval(model, eval_loader, total, m)
            acur_eval.append(acur)
            loss_eval.append(losses)
            BL_eval[:, :, tt] = np.squeeze(BL)


            # save the results  為了方便在每個時間查看
            mat_name = 'test_' + info + '.mat'
            sio.savemat(mat_name, {'acur_eval': acur_eval,
                                   'loss_eval': loss_eval,
                                   'BL_eval': BL_eval,
                                   'predict_beam': predict_beam,
                                   'true_beam': true_beam})


if __name__ == '__main__':
    execution_time = timeit.timeit(main, number=1)
    days, remainder = divmod(int(execution_time), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("程式執行時間：{} 天 {} 小時 {} 分鐘".format(days, hours, minutes))