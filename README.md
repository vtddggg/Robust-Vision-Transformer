# RVT: Rethinking the Design Principles of Robust Vision Transformer

***Note: Since the model is trained on our private platform, this transferred code has not been tested and may have some bugs. If you meet any problems, feel free to open an issue!***

This repository contains PyTorch code for Robust Vision Transformers.

![RVT](RVT.png)

For details see [Rethinking the Design Principles of Robust Vision Transformer](https://arxiv.org/abs/2105.07926) by Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Shaokai Ye, Yuan He and Hui Xue. 

# Usage

First, clone the repository locally:
```
git clone https://github.com/vtddggg/Robust-Vision-Transformer.git
```
Then, install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

We use 4 nodes with 8 gpus to train `RVT-Ti`, `RVT-S` and `RVT-B`:
## Training RVT-Ti

```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 main.py --model rvt_tiny --data-path /path/to/imagenet --output_dir output --dist-eval
```

## Training RVT-S

```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 main.py --model rvt_small --data-path /path/to/imagenet --output_dir output --dist-eval
```

## Training RVT-B

```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 main.py --model rvt_base --data-path /path/to/imagenet --output_dir output --batch-size 32 --dist-eval
```

If you want to train `RVT-Ti*`, `RVT-S*` or `RVT-B*`, simply add `--use_mask` and `--use_patch_aug` to enable positon-aware attention scaling and patch-wise augmentation.
