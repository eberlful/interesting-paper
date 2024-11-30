1. Cut Your Losses in Large-Vocabulary Language Models
2. Neural Metamorphosis
3. LAuReL: Learned Augmented Residual Layer


## Cut Your Losses in Large-Vocabulary Language Models

Github: https://github.com/apple/ml-cross-entropy?utm_source=catalyzex.com

Paper: https://arxiv.org/abs/2411.09009

Date: 13.11.2024

##### Abstract
As language models grow ever larger, so do their vocabularies. This has shifted the memory footprint of LLMs during training disproportionately to one single layer: the cross-entropy in the loss computation. Cross-entropy builds up a logit matrix with entries for each pair of input tokens and vocabulary items and, for small models, consumes an order of magnitude more memory than the rest of the LLM combined. We propose Cut Cross-Entropy (CCE), a method that computes the cross-entropy loss without materializing the logits for all tokens into global memory. Rather, CCE only computes the logit for the correct token and evaluates the log-sum-exp over all logits on the fly. We implement a custom kernel that performs the matrix multiplications and the log-sum-exp reduction over the vocabulary in flash memory, making global memory consumption for the cross-entropy computation negligible. This has a dramatic effect. Taking the Gemma 2 (2B) model as an example, CCE reduces the memory footprint of the loss computation from 24 GB to 1 MB, and the total training-time memory consumption of the classifier head from 28 GB to 1 GB. To improve the throughput of CCE, we leverage the inherent sparsity of softmax and propose to skip elements of the gradient computation that have a negligible (i.e., below numerical precision) contribution to the gradient. Experiments demonstrate that the dramatic reduction in memory consumption is accomplished without sacrificing training speed or convergence.

## Neural Metamorphosis

Github: https://adamdad.github.io/neumeta/

Paper: https://arxiv.org/pdf/2410.11878

Date: 10.10.2024

##### Abstract
This paper introduces a new learning paradigm termed Neural Metamorphosis (NeuMeta), which aims to build self-morphable neural networks. Contrary to crafting separate models for different architectures or sizes, NeuMeta directly learns the continuous weight manifold of neural networks. Once trained, we can sample weights for any-sized network directly from the manifold, even for previously unseen configurations, without retraining. To achieve this ambitious goal, NeuMeta trains neural implicit functions as hypernetworks. They accept coordinates within the model space as input, and generate corresponding weight values on the manifold. In other words, the implicit function is learned in a way, that the predicted weights is well-performed across various models sizes. In training those models, we notice that, the final performance closely relates on smoothness of the learned manifold. In pursuit of enhancing this smoothness, we employ two strategies. First, we permute weight matrices to achieve intra-model smoothness, by solving the Shortest Hamiltonian Path problem. Besides, we add a noise on the input coordinates when training the implicit function, ensuring models with various sizes shows consistent outputs. As such, NeuMeta shows promising results in synthesizing parameters for various network configurations. Our extensive tests in image classification, semantic segmentation, and image generation reveal that NeuMeta sustains full-size performance even at a 75% compression rate.

## LAuReL: Learned Augmented Residual Layer

Paper: https://arxiv.org/abs/2411.07501

Date: 13.11.2024

##### Abstract
One of the core pillars of efficient deep learning methods is architectural improvements such as the residual/skip connection, which has led to significantly better model convergence and quality. Since then the residual connection has become ubiquitous in not just convolutional neural networks but also transformer-based architectures, the backbone of LLMs.
In this paper we introduce \emph{Learned Augmented Residual Layer} (LAuReL) -- a novel generalization of the canonical residual connection -- with the goal to be an in-situ replacement of the latter while outperforming on both model quality and footprint metrics. Our experiments show that using \laurel can help boost performance for both vision and language models. For example, on the ResNet-50, ImageNet 1K task, it achieves 60% of the gains from adding an extra layer, while only adding 0.003% more parameters, and matches it while adding 2.6× fewer parameters.
