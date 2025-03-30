1. Cosmos World Foundation Model Platform for Physical AI
2. Intuitive physics understanding emerges from self-supervised pretraining on natural videos
3. PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?


## Cosmos World Foundation Model Platform for Physical AI

Github: https://github.com/NVIDIA/Cosmos

Paper: https://arxiv.org/abs/2501.03575

Date: 07.01.2024

##### Abstract
Physical AI needs to be trained digitally first. It needs a digital twin of itself, the policy model, and a digital twin of the world, the world model. In this paper, we present the Cosmos World Foundation Model Platform to help developers build customized world models for their Physical AI setups. We position a world foundation model as a general-purpose world model that can be fine-tuned into customized world models for downstream applications. Our platform covers a video curation pipeline, pre-trained world foundation models, examples of post-training of pre-trained world foundation models, and video tokenizers. To help Physical AI builders solve the most critical problems of our society, we make our platform open-source and our models open-weight with permissive licenses available via https://github.com/NVIDIA/Cosmos.

## Intuitive physics understanding emerges from self-supervised pretraining on natural videos

Github: https://github.com/facebookresearch/jepa-intuitive-physics

Paper: https://arxiv.org/abs/2502.11831

Date: 17.02.2025

##### Abstract
We investigate the emergence of intuitive physics understanding in general-purpose deep neural network models trained to predict masked regions in natural videos. Leveraging the violation-of-expectation framework, we find that video prediction models trained to predict outcomes in a learned representation space demonstrate an understanding of various intuitive physics properties, such as object permanence and shape consistency. In contrast, video prediction in pixel space and multimodal large language models, which reason through text, achieve performance closer to chance. Our comparisons of these architectures reveal that jointly learning an abstract representation space while predicting missing parts of sensory input, akin to predictive coding, is sufficient to acquire an understanding of intuitive physics, and that even models trained on one week of unique video achieve above chance performance. This challenges the idea that core knowledge -- a set of innate systems to help understand the world -- needs to be hardwired to develop an understanding of intuitive physics.

## PhysicsGen: Can Generative Models Learn from Images to Predict Complex Physical Relations?

Paper: https://arxiv.org/abs/2503.05333

Date: 07.03.2025

##### Abstract
The image-to-image translation abilities of generative learning models have recently made significant progress in the estimation of complex (steered) mappings between image distributions. While appearance based tasks like image in-painting or style transfer have been studied at length, we propose to investigate the potential of generative models in the context of physical simulations. Providing a dataset of 300k image-pairs and baseline evaluations for three different physical simulation tasks, we propose a benchmark to investigate the following research questions: i) are generative models able to learn complex physical relations from input-output image pairs? ii) what speedups can be achieved by replacing differential equation based simulations? While baseline evaluations of different current models show the potential for high speedups (ii), these results also show strong limitations toward the physical correctness (i). This underlines the need for new methods to enforce physical correctness. Data, baseline models and evaluation code http://www.physics-gen.org.
