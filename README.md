1. The folder includes the source codes of continuous-time beam tracking based on neural ordinary differential equation (ODE).
2. The folder includes the source codes of Continuous-Time mmWave Beam Prediction With ODE-LSTM Learning Architecture.
3. The folder is free for academic use, including dataset utilization, simulation result reproduction, model improvement, etc.
4. For academic use, the related work may be published in:
Kuang-Hao (Stanley) Liu, Huang-Chou Lin, ``Low Overhead Beam Alignment for Mobile Millimeter Channel Based on Continuous-Time Prediction'', minor revision, IEEE WCL.
5. The following are the steps for generating training data by using the folder "data":
   1. Download DeepMIMO functions and the data files of Raytracing scenarios O1 in 28 GHz operating frequency from the DeepMIMO website : https://www.deepmimo.net/
   2. parameters : set the DeepMIMO simulation parmeters.
   3. DeepMIMO_Dataset_Generator :  generate the millimeter wavew channel.
   4. generator_ODE_beam_tracking : generate User trajectory with prediction instants \tau = {0.1, 0.2, ..., 0.9} in each preciction periodic, predction duration is 1-secind predction duration .
   5. generator_ODE_beam_tracking_R1 : generate User trajectory with \tau = {0.01, 0.02, ..., 0.99} in each preciction periodic, predction duration is 1-secind.
   6. generator_ODE_beam_tracking_v2 : generate User trajectory with random \tau in each preciction periodic, predction duration is 1-secind.
   7. generator_ODE_beam_tracking_final : generate User trajectory with \tau = {0.01, 0.02, ..., 0.99}, predction duration is 4-secinds.
7. The 'beam_tracking_1s' folder contains three folders with different numbers of CNN layers, and each folder is compatible with strategies 1 through 3. The following provides the purpose of each file.
   1. model_ODE_few : neural network architecture and parameter tuning.
   2. train_dataloader_3D : load training data and batch output
   3. eval_dataloader_3D : load validation data and batch output 
   4. test_dataloader_3D : load testing data and batch output
   5. train_ODE_few : the main program for traing the predction model.
   6. test_ODE_few : the program for testing the predction model.
9.  The folder named "switching_mode_4s" contains 
10.  The 'benchmark' folder contains ARIMA, EKF, LSTM, and ODE-LSTM models. With the exception of ODE-LSTM using signals through 64 beams, each training model is simulated with both a signal through 11beams and 64 beams.
