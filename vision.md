1. DINOv2: Learning Robust Visual Features without Supervision
2. Towards Generalized Entropic Sparsification for Convolutional Neural Networks


## DINOv2: Learning Robust Visual Features without Supervision

Github: https://github.com/facebookresearch/dinov2

Paper: https://arxiv.org/pdf/2304.07193

Date: 02.02.2024

##### Abstract
The recent breakthroughs in natural language processing for model pretraining on large quantities of data have opened the way for similar foundation models in computer vision. These models could greatly simplify the use of images in any system by producing all-purpose visual features, i.e., features that work across image distributions and tasks without finetuning. This work shows that existing pretraining methods, especially self-supervised methods, can produce such features if trained on enough curated data from diverse sources. We revisit existing approaches and combine different techniques to scale our pretraining in terms of data and model size. Most of the technical contributions aim at accelerating and stabilizing the training at scale. In terms of data, we propose an automatic pipeline to build a dedicated, diverse, and curated image dataset instead of uncurated data, as typically done in the self-supervised literature. In terms of models, we train a ViT model (Dosovitskiy et al., 2020) with 1B parameters and distill it into a series of smaller models that surpass the best available all-purpose features, OpenCLIP (Ilharco et al., 2021) on most of the benchmarks at image and pixel levels.

## Towards Generalized Entropic Sparsification for Convolutional Neural Networks

Github: 

Paper: https://arxiv.org/abs/2404.04734

Date: 06.04.2024

##### Abstract
Convolutional neural networks (CNNs) are reported to be overparametrized. The search for optimal (minimal) and sufficient architecture is an NP-hard problem as the hyperparameter space for possible network configurations is vast. Here, we introduce a layer-by-layer data-driven pruning method based on the mathematical idea aiming at a computationally-scalable entropic relaxation of the pruning problem. The sparse subnetwork is found from the pre-trained (full) CNN using the network entropy minimization as a sparsity constraint. This allows deploying a numerically scalable algorithm with a sublinear scaling cost. The method is validated on several benchmarks (architectures): (i) MNIST (LeNet) with sparsity 55%-84% and loss in accuracy 0.1%-0.5%, and (ii) CIFAR-10 (VGG-16, ResNet18) with sparsity 73-89% and loss in accuracy 0.1%-0.5%.
