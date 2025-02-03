1. DINOv2: Learning Robust Visual Features without Supervision
2. Towards Generalized Entropic Sparsification for Convolutional Neural Networks
3. DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding
4. FilterViT and DropoutViT
5. MobileMamba: Lightweight Multi-Receptive Visual Mamba Network
6. Deformable DETR: Deformable Transformers for End-to-End Object Detection
7. Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion
8. iFormer: Integrating ConvNet and Transformer for Mobile Application


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

## DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding

Github: https://github.com/IDEA-Research/DINO-X-API

Paper: https://arxiv.org/pdf/2411.14347

Date: 22.11.2024

##### Abstract
In this paper, we introduce DINO-X, which is a unified object-centric vision model developed by IDEA Research with the best open-world object detection performance to date. DINO-X employs the same Transformer-based encoder-decoder architecture as Grounding DINO 1.5 to pursue an object-level representation for open-world object understanding. To make long-tailed object detection easy, DINO-X extends its input options to support text prompt, visual prompt, and customized prompt. With such flexible prompt options, we develop a universal object prompt to support prompt-free open-world detection, making it possible to detect anything in an image without requiring users to provide any prompt. To enhance the model's core grounding capability, we have constructed a large-scale dataset with over 100 million high-quality grounding samples, referred to as Grounding-100M, for advancing the model's open-vocabulary detection performance. Pre-training on such a large-scale grounding dataset leads to a foundational object-level representation, which enables DINO-X to integrate multiple perception heads to simultaneously support multiple object perception and understanding tasks, including detection, segmentation, pose estimation, object captioning, object-based QA, etc. Experimental results demonstrate the superior performance of DINO-X. Specifically, the DINO-X Pro model achieves 56.0 AP, 59.8 AP, and 52.4 AP on the COCO, LVIS-minival, and LVIS-val zero-shot object detection benchmarks, respectively. Notably, it scores 63.3 AP and 56.5 AP on the rare classes of LVIS-minival and LVIS-val benchmarks, both improving the previous SOTA performance by 5.8 AP. Such a result underscores its significantly improved capacity for recognizing long-tailed objects.

## DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding

Github: https://github.com/IDEA-Research/DINO-X-API

Paper: https://arxiv.org/pdf/2411.14347

Date: 21.11.2024

##### Abstract
In this paper, we introduce DINO-X, which is a unified object-centric vision model developed by IDEA Research with the best open-world object detection performance to date. DINO-X employs the same Transformer-based encoder-decoder architecture as Grounding DINO 1.5 to pursue an object-level representation for open-world object understanding. To make long-tailed object detection easy, DINO-X extends its input options to support text prompt, visual prompt, and customized prompt. With such flexible prompt options, we develop a universal object prompt to support prompt-free open-world detection, making it possible to detect anything in an image without requiring users to provide any prompt. To enhance the model's core grounding capability, we have constructed a large-scale dataset with over 100 million high-quality grounding samples, referred to as Grounding-100M, for advancing the model's open-vocabulary detection performance. Pre-training on such a large-scale grounding dataset leads to a foundational object-level representation, which enables DINO-X to integrate multiple perception heads to simultaneously support multiple object perception and understanding tasks, including detection, segmentation, pose estimation, object captioning, object-based QA, etc. Experimental results demonstrate the superior performance of DINO-X. Specifically, the DINO-X Pro model achieves 56.0 AP, 59.8 AP, and 52.4 AP on the COCO, LVIS-minival, and LVIS-val zero-shot object detection benchmarks, respectively. Notably, it scores 63.3 AP and 56.5 AP on the rare classes of LVIS-minival and LVIS-val benchmarks, both improving the previous SOTA performance by 5.8 AP. Such a result underscores its significantly improved capacity for recognizing long-tailed objects.

## FilterViT and DropoutViT

Github: https://github.com/BobSun98/FilterVIT

Paper: https://arxiv.org/abs/2410.22709

Date: 10.11.2024

##### Abstract
In this study, we introduce an enhanced version of ViT that conducts attention-based QKV operations during the initial stages of downsampling. Performing attention directly on high-resolution feature maps is computationally demanding due to the large size and numerous tokens. To mitigate this, we propose a filter attention mechanism that uses a Filter Block to create a salient mask (Filter Mask) for selecting the most informative pixels for attention.
The Filter Block scores the pixels of the feature map, and we sort these scores to retain only the top K pixels (with K varying across layers). This approach effectively decreases the number of tokens involved in the attention computation, reducing computational complexity and boosting processing speed. Furthermore, the salient mask provides interpretability, as the model focuses on regions of the image most critical to the outcome.
Our experimental results show that this model improves parameter efficiency and computational speed while enhancing accuracy. Compared to existing models, our approach significantly reduces resource consumption while maintaining high performance.

## MobileMamba: Lightweight Multi-Receptive Visual Mamba Network

Github: https://github.com/lewandofskee/MobileMamba

Paper: https://arxiv.org/abs/2411.15941

Date: 24.11.2024

##### Abstract
Previous research on lightweight models has primarily focused on CNNs and Transformer-based designs. CNNs, with their local receptive fields, struggle to capture long-range dependencies, while Transformers, despite their global modeling capabilities, are limited by quadratic computational complexity in high-resolution scenarios. Recently, state-space models have gained popularity in the visual domain due to their linear computational complexity. Despite their low FLOPs, current lightweight Mamba-based models exhibit suboptimal throughput. In this work, we propose the MobileMamba framework, which balances efficiency and performance. We design a three-stage network to enhance inference speed significantly. At a fine-grained level, we introduce the Multi-Receptive Field Feature Interaction(MRFFI) module, comprising the Long-Range Wavelet Transform-Enhanced Mamba(WTE-Mamba), Efficient Multi-Kernel Depthwise Convolution(MK-DeConv), and Eliminate Redundant Identity components. This module integrates multi-receptive field information and enhances high-frequency detail extraction. Additionally, we employ training and testing strategies to further improve performance and efficiency. MobileMamba achieves up to 83.6% on Top-1, surpassing existing state-of-the-art methods which is maximum x21 faster than LocalVim on GPU. Extensive experiments on high-resolution downstream tasks demonstrate that MobileMamba surpasses current efficient models, achieving an optimal balance between speed and accuracy.

## Deformable DETR: Deformable Transformers for End-to-End Object Detection

Github: https://github.com/fundamentalvision/Deformable-DETR

Paper: https://arxiv.org/abs/2010.04159

Date: 18.03.2021

##### Abstract
DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10 times less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach. Code is released at this https URL.

## Florence-VL: Enhancing Vision-Language Models with Generative Vision Encoder and Depth-Breadth Fusion

Github: https://github.com/JiuhaiChen/Florence-VL

Paper: https://arxiv.org/pdf/2412.04424

Date: 05.12.2024

##### Abstract
We present Florence-VL, a new family of multimodal large language models (MLLMs) with enriched visual representations produced by Florence-2, a generative vision foundation model. Unlike the widely used CLIP-style vision transformer trained by contrastive learning, Florence-2 can capture different levels and aspects of visual features, which are more versatile to be adapted to diverse downstream tasks. We propose a novel feature-fusion architecture and an innovative training recipe that effectively integrates Florence-2's visual features into pretrained LLMs, such as Phi 3.5 and LLama 3. In particular, we propose "depth-breath fusion (DBFusion)" to fuse the visual features extracted from different depths and under multiple prompts. Our model training is composed of end-to-end pretraining of the whole model followed by finetuning of the projection layer and the LLM, on a carefully designed recipe of diverse open-source datasets that include high-quality image captions and instruction-tuning pairs. Our quantitative analysis and visualization of Florence-VL's visual features show its advantages over popular vision encoders on vision-language alignment, where the enriched depth and breath play important roles. Florence-VL achieves significant improvements over existing state-of-the-art MLLMs across various multi-modal and vision-centric benchmarks covering general VQA, perception, hallucination, OCR, Chart, knowledge-intensive understanding, etc. To facilitate future research, our models and the complete training recipe are open-sourced. https://github.com/JiuhaiChen/Florence-VL

## VisionZip: Longer is Better but Not Necessary in Vision Language Models

Github: https://github.com/dvlab-research/VisionZip

Paper: https://arxiv.org/pdf/2412.04467

Date: 05.12.2024

##### Abstract
Recent advancements in vision-language models have enhanced performance by increasing the length of visual tokens, making them much longer than text tokens and significantly raising computational costs. However, we observe that the visual tokens generated by popular vision encoders, such as CLIP and SigLIP, contain significant redundancy. To address this, we introduce VisionZip, a simple yet effective method that selects a set of informative tokens for input to the language model, reducing visual token redundancy and improving efficiency while maintaining model performance. The proposed VisionZip can be widely applied to image and video understanding tasks and is well-suited for multi-turn dialogues in real-world scenarios, where previous methods tend to underperform. Experimental results show that VisionZip outperforms the previous state-of-the-art method by at least 5% performance gains across nearly all settings. Moreover, our method significantly enhances model inference speed, improving the prefilling time by 8x and enabling the LLaVA-Next 13B model to infer faster than the LLaVA-Next 7B model while achieving better results. Furthermore, we analyze the causes of this redundancy and encourage the community to focus on extracting better visual features rather than merely increasing token length. Our code is available at https://github.com/dvlab-research/VisionZip .

## iFormer: Integrating ConvNet and Transformer for Mobile Application

Github: https://github.com/ChuanyangZheng/iFormer

Paper: https://arxiv.org/abs/2501.15369

Date: 26.01.2025

##### Abstract
We present a new family of mobile hybrid vision networks, called iFormer, with a focus on optimizing latency and accuracy on mobile applications. iFormer effectively integrates the fast local representation capacity of convolution with the efficient global modeling ability of self-attention. The local interactions are derived from transforming a standard convolutional network, i.e., ConvNeXt, to design a more lightweight mobile network. Our newly introduced mobile modulation attention removes memory-intensive operations in MHA and employs an efficient modulation mechanism to boost dynamic global representational capacity. We conduct comprehensive experiments demonstrating that iFormer outperforms existing lightweight networks across various tasks. Notably, iFormer achieves an impressive Top-1 accuracy of 80.4\% on ImageNet-1k with a latency of only 1.10 ms on an iPhone 13, surpassing the recently proposed MobileNetV4 under similar latency constraints. Additionally, our method shows significant improvements in downstream tasks, including COCO object detection, instance segmentation, and ADE20k semantic segmentation, while still maintaining low latency on mobile devices for high-resolution inputs in these scenarios.
