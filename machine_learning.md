1. Cut Your Losses in Large-Vocabulary Language Models
2. Neural Metamorphosis
3. LAuReL: Learned Augmented Residual Layer
4. Differentiable Weightless Neural Networks
5. DeMo: Decoupled Momentum Optimization
6. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training


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

## Differentiable Weightless Neural Networks

Github: https://github.com/alanbacellar/DWN

Paper: https://arxiv.org/abs/2410.11112

Date: 27.11.2024

##### Abstract
We introduce the Differentiable Weightless Neural Network (DWN), a model based on interconnected lookup tables. Training of DWNs is enabled by a novel Extended Finite Difference technique for approximate differentiation of binary values. We propose Learnable Mapping, Learnable Reduction, and Spectral Regularization to further improve the accuracy and efficiency of these models. We evaluate DWNs in three edge computing contexts: (1) an FPGA-based hardware accelerator, where they demonstrate superior latency, throughput, energy efficiency, and model area compared to state-of-the-art solutions, (2) a low-power microcontroller, where they achieve preferable accuracy to XGBoost while subject to stringent memory constraints, and (3) ultra-low-cost chips, where they consistently outperform small models in both accuracy and projected hardware area. DWNs also compare favorably against leading approaches for tabular datasets, with higher average rank. Overall, our work positions DWNs as a pioneering solution for edge-compatible high-throughput neural networks.

## DeMo: Decoupled Momentum Optimization

Github: https://github.com/bloc97/DeMo

Paper: https://arxiv.org/abs/2411.19870

Date: 29.11.2024

##### Abstract
Training large neural networks typically requires sharing gradients between accelerators through specialized high-speed interconnects. Drawing from the signal processing principles of frequency decomposition and energy compaction, we demonstrate that synchronizing full optimizer states and model parameters during training is unnecessary. By decoupling momentum updates and allowing controlled divergence in optimizer states across accelerators, we achieve improved convergence compared to state-of-the-art optimizers. We introduce {\textbf{De}}coupled {\textbf{Mo}}mentum (DeMo), a fused optimizer and data parallel algorithm that reduces inter-accelerator communication requirements by several orders of magnitude. This enables training of large neural networks even with limited network bandwidth and heterogeneous hardware. Our method is topology-agnostic and architecture-independent and supports scalable clock-synchronous distributed training with negligible compute and memory overhead. Empirical results show that models trained with DeMo match or exceed the performance of equivalent models trained with AdamW, while eliminating the need for high-speed interconnects when pre-training large scale foundation models. An open source reference PyTorch implementation is published on GitHub at this https URL

## SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training

Paper: https://arxiv.org/abs/2501.17161

Date: 28.01.2025

##### Abstract
Supervised fine-tuning (SFT) and reinforcement learning (RL) are widely used post-training techniques for foundation models. However, their roles in enhancing model generalization capabilities remain unclear. This paper studies the difference between SFT and RL on generalization and memorization, focusing on text-based rule variants and visual variants. We introduce GeneralPoints, an arithmetic reasoning card game, and adopt V-IRL, a real-world navigation environment, to assess how models trained with SFT and RL generalize to unseen variants in both textual and visual domains. We show that RL, especially when trained with an outcome-based reward, generalizes across both rule-based textual and visual variants. SFT, in contrast, tends to memorize training data and struggles to generalize out-of-distribution scenarios. Further analysis reveals that RL improves the model's underlying visual recognition capabilities, contributing to its enhanced generalization in the visual domain. Despite RL's superior generalization, we show that SFT remains essential for effective RL training; SFT stabilizes the model's output format, enabling subsequent RL to achieve its performance gains. These findings demonstrates the capability of RL for acquiring generalizable knowledge in complex, multi-modal tasks.
