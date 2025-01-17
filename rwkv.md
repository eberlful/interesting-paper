1. Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence

## Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence

Paper: https://arxiv.org/abs/2404.05892

Github: 08.2024

##### Abstract
We present Eagle (RWKV-5) and Finch (RWKV-6), sequence models improving upon the RWKV (RWKV-4) architecture. Our architectural design advancements include multi-headed matrix-valued states and a dynamic recurrence mechanism that improve expressivity while maintaining the inference efficiency characteristics of RNNs. We introduce a new multilingual corpus with 1.12 trillion tokens and a fast tokenizer based on greedy matching for enhanced multilinguality. We trained four Eagle models, ranging from 0.46 to 7.5 billion parameters, and two Finch models with 1.6 and 3.1 billion parameters and find that they achieve competitive performance across a wide variety of benchmarks. We release all our models on HuggingFace under the Apache 2.0 license. Models at: this https URL Training code at: this https URL Inference code at: this https URL Time-parallel training code at: this https URL
