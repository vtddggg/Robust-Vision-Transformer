# RVT: Robust Vision Transformers

This repository contains PyTorch code for Robust Vision Transformers.

They obtain competitive tradeoffs in terms of speed / precision:

![RVT](.github/RVT.png)

For details see [Rethinking the Design Principles of Robust Vision Transformer](https://arxiv.org/abs/2012.12877) by Xiaofeng Mao, Gege Qi, Yuefeng Chen, Yuan He and Hui Xue. 

If you use this code for a paper please cite:

```
@article{touvron2020deit,
  title={Training data-efficient image transformers & distillation through attention},
  author={Hugo Touvron and Matthieu Cord and Matthijs Douze and Francisco Massa and Alexandre Sablayrolles and Herv\'e J\'egou},
  journal={arXiv preprint arXiv:2012.12877},
  year={2020}
}
```

# Usage

First, clone the repository locally:
```
git clone https://github.com/facebookresearch/deit.git
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
