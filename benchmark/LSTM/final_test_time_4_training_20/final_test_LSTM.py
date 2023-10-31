# the trained models can be downloaded from the link https://cloud.tsinghua.edu.cn/d/e3b3793cb4ed4950be66/.

import torch.optim as optim
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
# dataloader for testing
from test_dataloader_3D import test_Dataloader_3D


import sys

# model_evaluation
def eval(model, loader, total, m):
    # input: trained model, test data loader, total (number of total points) and m (points to be predicted between two times of beam training)
    # output: accuracy, losses and normalized beamforming gain
    # reset dataloader
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    BL = np.zeros((40, m))
    BL_sample = np.zeros((40, m, 2560))
    # running loss
    running_loss = 0
    # count batch number
    batch_num = 0
    batch_size = 32
    # count predicted instants between two times of beam training

    # evaluate validation set
    while not done:
        # read files
        # channels: sequence of mmWave beam training received signal vectors
        # labels: sequence of optimal mmWave beam indices
        # beam power: sequence of mmWave beam training received signal power vectors
        channels, labels, beam_power, done, count = loader.next_batch()
        # if nothing is loaded, break
        if labels.shape[0] == 0:
            break;

        labels = labels.to(torch.int64)
        # select data for beam training
        channel_train = channels[:, :, 0 : total : (m + 1), :]

        if count == True:
            batch_num += 1
            # predicted results
            # out_tensor.shape: pre_points * length * batch_size * num_of_beam
            out_tensor = model(channel_train, pre_points = m)
            # calculate loss function
            loss = 0
            # for all predictions after 10 beam trainings
            for loss_count in range(40):
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
    print("batch_num: %.3f" % (batch_num))
    print("Accuracy: %.3f" % (acur))
    print("Loss: %.3f" % (losses))
    print("Beam power loss:")
    print(BL.T)

    return acur, losses, BL


# main function for model training and evaluation
# output: accuracy, losses and normalized beamforming gain
def main():
    # first loop for different velocities
    for velocity in [30]:
        # save corresponding information
        print("velocity:", velocity)
        version_name = 'LSTM_ir_3CNN_1LSTM_v2'
        # a = 0.2v m/s^2
        info = 'WCL_v' + str(velocity) + '_a' + str(velocity * 0.2) + '_25dBm_' + version_name
        print(info)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        t = 1  # training time
        batch_size = 32 # batch size
        total = 4000 # total points
        m = 99 # the number of points to be predicted between two times of beam training
        print('batch_size:%d' % (batch_size))

        path = r'C:\Users\Aiden\Desktop\my_data\time_01_SNR_104\final_4s\ODE_dataset(final_test)_v' + str(velocity)

        eval_loader = test_Dataloader_3D(path=path, batch_size=batch_size, device=device)

        # save results
        acur_eval = []
        loss_eval = []
        BL_eval = np.zeros((40, m, t))  # normalized beamforming gain

        # first loop for training runnings
        for tt in range(t):
            print('Train %d times' % (tt))
            # load model
            model_name = info + '_' + str(tt) + '_MODEL.pkl'
            model = torch.load(model_name)
            model.to(device)

            # print parameters
            for name, param in model.named_parameters():
                print('Name:', name, 'Size:', param.size())

            # evaluate the model
            model.eval()
            print('the evaling set:')

            acur, losses, BL = eval(model, eval_loader, total, m)
            acur_eval.append(acur)
            loss_eval.append(losses)
            BL_eval[:, :, tt] = np.squeeze(BL)

            # save the results
            mat_name = 'test_' + info + '.mat'
            sio.savemat(mat_name, {'acur_eval': acur_eval,
                                   'loss_eval': loss_eval,
                                    'BL_eval': BL_eval})

if __name__ == '__main__':
    main()