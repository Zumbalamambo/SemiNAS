# SemiNAS: Semi-Supervised Neural Architecture Search

This repository contains the code used for SemiNAS on TTS.


## Environments and Requirements
The code is built and tested on Pytorch 1.2

First, install dependent python packages via:
```
pip install -r requirements.txt
```
Then, install libsndfile1:
```
sudo apt-get install libsndfile1
```
Finally, download dataset in [`Google Drive`](https://drive.google.com/open?id=1EBWactIa4MTrsf7yyBWpnZvanHOwfzOp), and unzip it to the current folder:
```
unzip data.zip
```

## Searching Architectures
To run the search process, please refer to `runs/search_for_low_resource.sh` and `runs/search_for_robustness.sh` which are for low-resource setting and robustness setting respectively. It requires 4 GPU cards to train.
```
cd runs
bash search_for_low_resource.sh
bash search_for_robust.sh
cd ..
```
After it finishes, you can find the final discovored top architectures in the $OUTPUT_DIR which in the scripts are `outputs/search_for_low_resource` and `outputs/search_for_robustness` respectively. The final discovered top architectures are listed in the file `arch_pool.4` where each line is an architecture represented in sequence which is further used in the final training.

## Training Architectures

### Training the architecture discovered by SemiNAS
To directly train the architecture discovered by SemiNAS as we report in the paper, please refer to `runs/train_low_resource.sh` for low-resource setting and `runs/train_robustness.sh` for robustness setting. It requires 4 GPU cards to train.
```
cd runs
bash train_low_resource.sh
bash train_robustness.sh
cd ..
```

### Training customized architectures
To train a customized architecture (e.g., discovered by SemiNAS of your run), you can modify the script `runs/train_low_resource.sh` for low-resource setting and `runs/train_robustness.sh` for robustness setting by replacing the `--arch` with your customized architecture sequence.


## Evaluating the architecture discovered by SemiNAS in our paper
We provide the checkpoints of low-resource setting and robustness setting in [`Google Dirve`](https://drive.google.com/open?id=1qCRJvy_rTHSgj5XYXLeJfgE4Eheniz61)
You can download it and unzip it to the current path:
```
unzip checkpoints.zip
```
To evalute it, please refer to `runs/test_low_resource.sh` for low-resource setting and `runs/test_robustness.sh` for robustness setting. 
```
cd runs
bash test_low_resource.sh
bash test_robustness.sh
cd ..
```
The generated audio files are in `$OUTPUT_DIR/generated__best` where `OUTPUT_DIR` is specified in the scripts.