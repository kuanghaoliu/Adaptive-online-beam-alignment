This repository contains the source codes for reproducing the results in the following paper
Kuang-Hao (Stanley) Liu, Huang-Chou Lin, "Low Overhead Beam Alignment for Mobile Millimeter Channel Based on Continuous-Time Prediction"

# Generate training data
Here are the steps for generating training data. The required files can be found in the folder "data".
   1. Download DeepMIMO and the data files of Raytracing scenario O1 in 28 GHz operating frequency from the DeepMIMO website : https://www.deepmimo.net/
   2. Set the simulation in *parameters.m*.
   3. Generate the millimeter wave channel using *DeepMIMO_Dataset_Generator.m*.
   4. Generate User trajectory at the normalized prediction instant $\tau$ with four settings.
      - *generator_ODE_beam_tracking_v2.m*: $\tau$ is randomly distributed within each prediction period. This is used to train the model.
      - *generator_ODE_beam_tracking.m*: $\tau = 0.1, 0.2, \cdots, 0.9$. This is to plot the simulation result.
      - *generator_ODE_beam_tracking_R1.m*: $\tau = 0.01, 0.02, ..., 0.99$. This is used to plot the simulation result.
      - *generator_ODE_beam_tracking_final.m*: $\tau = 0.01, 0.02, ..., 0.99$ for each trajectory lasting for 4 seconds. A longer duration is used to examine the prediction performance when mode switching is enabled (see [Mode switching enabled](#Mode switching enabled)).

# Mode switching disabled (see folder *mode switching disabled*)
This is the case where beam training is performed every $$T$$ seconds, where $T=100$ ms by default. Each trajectory lasts for 1 second.
- train_ODE_few.py: the main program for training the model.
- test_ODE_few.py: the program for testing the model.

Supporting files
- model_ODE_few.py: neural network architecture and parameter tuning.
- train_dataloader_3D.py: load training data and batch output
- eval_dataloader_3D.py: load validation data and batch output 
- test_dataloader_3D.py: load testing data and batch output
  
# Mode switching enabled (see folder *switching mode_4s*)
This is the case where mode switching is enabled every $T$ seconds, where $T=100$ ms by default. Each trajectory lasts for 4 seconds. Except the longer moving trajectory, the model is the same as the one used in the folder *Mode switching disabled*. 
- test__ODE_final_adaptive switching.py: the program for testing the **adaptive** switching mode between beam scanning and beam tracking. 
- test__ODE_final_periodic switching.py: the program for testing the **periodic** switching mode between beam scanning and beam tracking.

  Supporting files
- test__ODE_final_11beam.py: use this file if you want to test the performance of beam tracking mode using 11 probing beams.
- test__ODE_final_64beam.py: use this file if you want to test the performance of beam scanning mode using 64 probing beams.
- test__ODE_final_64beam_dif_seq.py: use this file if want to test the performance of beam scanning mode using different sequence in LSTM.
- model_ODE.py: neural network architecture and parameter tuning for beam scanning.
- model_ODE_few.py: neural network architecture and parameter tuning for beam tracking.
- test_dataloader_3D.py: load testing data and batch output

# Benchmark schemes
The folder **benchmark** contains the files for implementing ARIMA, EKF, LSTM, and ODE-LSTM. For a fair comparison, mode switching is enabled for ARIMA, EKF, and LSTM. ODE-LSTM proposed in reference [9] considers beam scanning only. 
