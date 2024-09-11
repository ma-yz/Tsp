# Towards Better Out-of-distribution Generalization for Model Pruning

This repo contains required scripts to reproduce results from paper:

Towards Better Out-of-distribution Generalization for Model Pruning<br>
Yuzhe Ma, Zhi Zhou, Yufeng Li.<br>


## Installation

### prepare environment

The experimental results are obtained in torch1.8.1+cu111 environment.

Reproduce the experimental environment:
```bash
conda create --name TSP python==3.7
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
cd /path/to/TSP
pip install -r requirements.txt
```

## Train a model for pruning
```bash
cd /path/to/TSP
python -m domainbed.scripts.train --data_dir /path/to/dataset --dataset PACS --algorithm CORAL --arch resnet50 --test_envs 0 --model_dir ./pretrain/resnet50_coral_pacs_test0
```

## prune the model
```bash
cd /path/to/TSP
python main.py --save_dir=prune_result/res50_coral_pacs1_50 --algorithm TSP --dataset PACS --data=/path/to/dataset --test_envs 1 --pruning_config=./configs/imagenet_resnet50_prune50.json --load_model=./pretrain/resnet50_coral_pacs_test0/model.pkl --epochs=30 --batch-size=32 --lr=1e-3 --model=resnet50 --mgpu=True --tensorboard=True --num_workers 8
```


Note: we recommend finetuning for 10 epochs when using TSP and set epochs 20 with other pruning algorithms. 

We provide config files for pruning of different compression ratios. Percentage means the ratio of activated parameters after pruning.

## Parameter description


| algorithm       |  description | 
| ------------- |-------------|
| TSP         |  Our two step pruning |
| baseline    | Taylor pruning |
| SWAD        | The average weight near the convergence point of the model |
| DG_pruning  | Add regularization term to select pruned parameter |

