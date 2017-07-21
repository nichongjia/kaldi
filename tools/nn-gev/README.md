# Neural network based GEV beamformer

## Introduction

This repository contains code to replicate the results for the 3rd CHiME challenge using a *NN-GEV* Beamformer.

## Install

This code requires Python 3 to run (although most parts should be compatible with Python 2.7). Install the necessary modules:

```
pip install chainer
pip install tqdm
pip install SciPy
pip install scikit-learn
pip install librosa
```

## Usage

  1. Extract the speech and noise images for the SimData using the modified Matlab script in CHiME3/tools/simulation
  2. Start the training for the BLSTM model using the GPU with id 0 and the data directory ``data``:
  
      ```
      python train.py --chime_dir=../chime/data --gpu 0 data BLSTM
      ```
      
      This will first create the training data (i.e. the binary mask targets) and then run the training with early stopping. Instead of ``BLSTM`` it is also possible to specify ``FW`` to train a simple feed-forward model.
      
  3. Start the beamforming:
  
    ```
    beamform.sh ../chime/data data/export_BLSTM data/BLSTM_model/best.nnet BLSTM
    ```
    
    This will apply the beamformer to every utterance of the CHiME database and store the resulting audio file in ``data/export_BLSTM``. The model ``data/BLSTM_model/best.nnet`` is used to generate the masks.
    
  4. Start the kaldi baseline using the exported data.

  If you want to use the beamformer with a different database, take a look at ``beamform.py`` and ``chime_data`` and modify it accordingly.

## Results
With the new baseline, you should get the following results:
  
    ```
    local/chime3_calc_wers.sh exp/tri3b_tr05_multi_open_gev open_gev
    compute dt05 WER for each location
    
    -------------------
    best overall dt05 WER 10.51% (language model weight = 12)
    -------------------
    dt05_simu WER: 10.58% (Average), 8.76% (BUS), 13.19% (CAFE), 9.57% (PEDESTRIAN), 10.80% (STREET)
    -------------------
    dt05_real WER: 10.44% (Average), 11.43% (BUS), 10.21% (CAFE), 9.31% (PEDESTRIAN), 10.81% (STREET)
    -------------------
    et05_simu WER: 12.43% (Average), 10.50% (BUS), 13.54% (CAFE), 12.46% (PEDESTRIAN), 13.24% (STREET)
    -------------------
    et05_real WER: 14.74% (Average), 17.72% (BUS), 14.98% (CAFE), 13.53% (PEDESTRIAN), 12.72% (STREET)
    -------------------

    local/chime3_calc_wers.sh exp/tri4a_dnn_tr05_multi_open_gev open_gev
    compute dt05 WER for each location
    
    -------------------
    best overall dt05 WER 7.86% (language model weight = 11)
    -------------------
    dt05_simu WER: 7.85% (Average), 6.18% (BUS), 10.01% (CAFE), 7.05% (PEDESTRIAN), 8.16% (STREET)
    -------------------
    dt05_real WER: 7.87% (Average), 8.94% (BUS), 8.44% (CAFE), 6.43% (PEDESTRIAN), 7.67% (STREET)
    -------------------
    et05_simu WER: 8.86% (Average), 7.53% (BUS), 9.62% (CAFE), 8.91% (PEDESTRIAN), 9.39% (STREET)
    -------------------
    et05_real WER: 11.46% (Average), 14.62% (BUS), 12.05% (CAFE), 9.59% (PEDESTRIAN), 9.60% (STREET)
    -------------------
    
    ./local/chime3_calc_wers_smbr.sh exp/tri4a_dnn_tr05_multi_open_gev_smbr_i1lats open_gev exp/tri4a_dnn_tr05_multi_open_gev/graph_tgpr_5k
    compute WER for each location
    
    -------------------
    best overall dt05 WER 7.12% (language model weight = 11)
     (Number of iterations = 4)
    -------------------
    dt05_simu WER: 7.00% (Average), 5.53% (BUS), 8.92% (CAFE), 6.46% (PEDESTRIAN), 7.08% (STREET)
    -------------------
    dt05_real WER: 7.25% (Average), 8.57% (BUS), 7.86% (CAFE), 5.77% (PEDESTRIAN), 6.80% (STREET)
    -------------------
    et05_simu WER: 8.11% (Average), 6.82% (BUS), 8.85% (CAFE), 7.96% (PEDESTRIAN), 8.80% (STREET)
    -------------------
    et05_real WER: 10.51% (Average), 13.01% (BUS), 10.93% (CAFE), 9.17% (PEDESTRIAN), 8.93% (STREET)
    -------------------
    
    local/chime3_calc_wers.sh exp/tri4a_dnn_tr05_open_gev_beamformer_smbr_lmrescore gev_beamformer_rnnlm_5k_h300_w0.5_n100
    compute dt05 WER for each location
    
    -------------------
    best overall dt05 WER 4.77% (language model weight = 12)
    -------------------
    dt05_simu WER: 5.01% (Average), 3.94% (BUS), 6.55% (CAFE), 4.35% (PEDESTRIAN), 5.21% (STREET)
    -------------------
    dt05_real WER: 4.53% (Average), 5.35% (BUS), 4.65% (CAFE), 3.64% (PEDESTRIAN), 4.47% (STREET)
    -------------------
    et05_simu WER: 5.60% (Average), 5.10% (BUS), 6.07% (CAFE), 5.86% (PEDESTRIAN), 5.36% (STREET)
    -------------------
    et05_real WER: 7.45% (Average), 9.55% (BUS), 7.94% (CAFE), 5.98% (PEDESTRIAN), 6.35% (STREET)
    -------------------
    ```
    
## Citation
  If you use this code for your experiments, please consider citing the following paper:
  
  ```
  @inproceedings{Hey2016,
  title = {NEURAL NETWORK BASED SPECTRAL MASK ESTIMATION FOR ACOUSTIC BEAMFORMING},
  author = {J. Heymann, L. Drude, R. Haeb-Umbach},
  year = {2016},
  date = {2016-03-20},
  booktitle = {Proc. IEEE Intl. Conf. on Acoustics, Speech and Signal Processing (ICASSP)},
  keywords = {},
  pubstate = {forthcoming},
  tppubtype = {inproceedings}
  }
  ```
