1. Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence
2. RWKV-UNet: Improving UNet with Long-Range Cooperation for Effective Medical Image Segmentation

## Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence

Paper: https://arxiv.org/abs/2404.05892

Github: 08.2024

##### Abstract
We present Eagle (RWKV-5) and Finch (RWKV-6), sequence models improving upon the RWKV (RWKV-4) architecture. Our architectural design advancements include multi-headed matrix-valued states and a dynamic recurrence mechanism that improve expressivity while maintaining the inference efficiency characteristics of RNNs. We introduce a new multilingual corpus with 1.12 trillion tokens and a fast tokenizer based on greedy matching for enhanced multilinguality. We trained four Eagle models, ranging from 0.46 to 7.5 billion parameters, and two Finch models with 1.6 and 3.1 billion parameters and find that they achieve competitive performance across a wide variety of benchmarks. We release all our models on HuggingFace under the Apache 2.0 license. Models at: this https URL Training code at: this https URL Inference code at: this https URL Time-parallel training code at: this https URL

## RWKV-UNet: Improving UNet with Long-Range Cooperation for Effective Medical Image Segmentation

Github: https://github.com/juntaoJianggavin/RWKV-UNet

Paper: https://arxiv.org/abs/2501.08458

Date: 14.01.2025

##### Abstract
In recent years, there have been significant advancements in deep learning for medical image analysis, especially with convolutional neural networks (CNNs) and transformer models. However, CNNs face limitations in capturing long-range dependencies while transformers suffer high computational complexities. To address this, we propose RWKV-UNet, a novel model that integrates the RWKV (Receptance Weighted Key Value) structure into the U-Net architecture. This integration enhances the model's ability to capture long-range dependencies and improve contextual understanding, which is crucial for accurate medical image segmentation. We build a strong encoder with developed inverted residual RWKV (IR-RWKV) blocks combining CNNs and RWKVs. We also propose a Cross-Channel Mix (CCM) module to improve skip connections with multi-scale feature fusion, achieving global channel information integration. Experiments on benchmark datasets, including Synapse, ACDC, BUSI, CVC-ClinicDB, CVC-ColonDB, Kvasir-SEG, ISIC 2017 and GLAS show that RWKV-UNet achieves state-of-the-art performance on various types of medical image segmentation. Additionally, smaller variants, RWKV-UNet-S and RWKV-UNet-T, balance accuracy and computational efficiency, making them suitable for broader clinical applications.




Information:
* https://fullstackdeeplearning.com/blog/posts/rwkv-explainer/#defining-internal-computation-and-propagation-gated-mlp-and-attention
* https://johanwind.github.io/2023/03/23/rwkv_details.html
* https://github.com/ridgerchu/SpikeGPT
* https://arxiv.org/abs/2305.13048
* https://github.com/BlinkDL/RWKV-LM
* https://arxiv.org/abs/2105.14103 (An Attention Free Transformer)
* https://www.rwkv.com/
