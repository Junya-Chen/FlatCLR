# FlatNCE: A Novel Contrastive Representation Learning Objective

This is the official code repository for the paper 
[Simpler, Faster, Stronger: Breaking The log-K Curse On Contrastive Learners With FlatNCE.](https://arxiv.org/pdf/2107.01152.pdf)

 InfoNCE-based contrastive representation learners, such as SimCLR, have been tremendously successful in recent years. However, these contrastive schemes are notoriously resource demanding, as their effectiveness breaks down with small-batch training (i.e., the log-K curse, whereas K is the batch-size). In this work, we reveal mathematically why contrastive learners fail in the small-batch-size regime, and present a novel simple, non-trivial contrastive objective named FlatNCE, which fixes this issue. Unlike InfoNCE, our FlatNCE no longer explicitly appeals to a discriminative classification goal for contrastive learning. Theoretically, we show FlatNCE is the mathematical dual formulation of InfoNCE, thus bridging the classical literature on energy modeling; and empirically, we demonstrate that, with minimal modification of code, FlatNCE enables immediate performance boost independent of the subject-matter engineering efforts. The significance of this work is furthered by the powerful generalization of contrastive learning techniques, and the introduction of new tools to monitor and diagnose contrastive training. We substantiate our claims with empirical evidence on CIFAR10, ImageNet, and other datasets, where FlatNCE consistently outperforms InfoNCE.

## Usage
To start training on the imagenet dataset (or any other), first download and decompress, and place it under ./datasets/imagenet. 
### Pretraining
We have faster version and slower version (SimCLR implementation) of data augmentation, and faster version only supports for cifar10 and cifar100.

To pretrain the SimCLR on CIFAR-10 with faster version, try the following command:
```
python main.py --dataset_name=cifar10 --clr=simclr --faster_version=True
```

To pretrain the FlatCLR on CIFAR-10 with normal version, try the following command:
```
python main.py --dataset_name=cifar10 --clr=flatclr --faster_version=False
```

To pretrain the SimCLR on Imagenet, try the following command:
```
python main.py --dataset_name=imagenet --clr=simclr
```

To pretrain the FlatCLR on Imagenet, try the following command:
```
python main.py --dataset_name=imagenet --clr=flatclr
```

The trained models are saved at: results/{dataset_name}/{batch_size}_SimCLR/{date_string}/checkpoint_{:04d}.pth.tar'.format(epochs)

Note that learning rate of 0.3 with learning_rate_scaling=linear is equivalent to that of 0.075 with learning_rate_scaling=sqrt when the batch size is 4096. However, using sqrt scaling allows it to train better when smaller batch size is used.

Quick Lookup:
| Batch size  | Linear scaling lr | Sqrt scaling lr
| ------------- | ------------- |-------------|
|128|lr=0.15|lr=0.85
|256|lr=0.3|lr=1.20
|512|lr=0.6|lr=1.70
|1024|lr=1.2|lr=2.40
|2048|lr=2.4|lr=3.39
|4096|lr=4.8|lr=4.80
|8192|lr=9.6|lr=6.79

### Resume Pretraining
e.g., To resume a flatclr model from 26 epochs:
```
python main.py --clr=flatclr --log_dir=results/imagenet/512_FlatCLR/01-06-2021-21-52-10 --train_from=checkpoint_0026.pth.tar
```
### Linear Classification
To train the linear classification on Imagenet, try the following command:
```
python main.py --dataset_name=imagenet --train_mode=eval --transfer_mode=linear_eval --checkpoint_dir=results/imagenet/512_FlatCLR/01-06-2021-21-52-10
```

### Finetune
To finetune the classifier, try the following command:
```
python main.py --dataset_name=imagenet --train_mode=eval --transfer_mode=finetune --checkpoint_dir=results/imagenet/512_FlatCLR/01-06-2021-21-52-10
```
## Citation
If you reference or use our method, code or results in your work, please consider citing the FlatNCE paper:
```
@article{chen2021simpler,
  title={Simpler, Faster, Stronger: Breaking The log-K Curse On Contrastive Learners With FlatNCE},
  author={Chen, Junya and Gan, Zhe and Li, Xuan and Guo, Qing and Chen, Liqun and Gao, Shuyang and Chung, Tagyoung and Xu, Yi and Zeng, Belinda and Lu, Wenlian and others},
  journal={arXiv preprint arXiv:2107.01152},
  year={2021}
}
```
