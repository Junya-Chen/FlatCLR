# FlatNCE: A Novel Contrastive Representation Learning Objective

This is the official code repository for the paper 
[Simpler, Faster, Stronger: Breaking The log-K Curse On Contrastive Learners With FlatNCE.](https://arxiv.org/pdf/2107.01152.pdf)

 InfoNCE-based contrastive representation learners, such as SimCLR, have been tremendously successful in recent years. However, these contrastive schemes are notoriously resource demanding, as their effectiveness breaks down with small-batch training (i.e., the log-K curse, whereas K is the batch-size). In this work, we reveal mathematically why contrastive learners fail in the small-batch-size regime, and present a novel simple, non-trivial contrastive objective named FlatNCE, which fixes this issue. Unlike InfoNCE, our FlatNCE no longer explicitly appeals to a discriminative classification goal for contrastive learning. Theoretically, we show FlatNCE is the mathematical dual formulation of InfoNCE, thus bridging the classical literature on energy modeling; and empirically, we demonstrate that, with minimal modification of code, FlatNCE enables immediate performance boost independent of the subject-matter engineering efforts. The significance of this work is furthered by the powerful generalization of contrastive learning techniques, and the introduction of new tools to monitor and diagnose contrastive training. We substantiate our claims with empirical evidence on CIFAR10, ImageNet, and other datasets, where FlatNCE consistently outperforms InfoNCE.

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
