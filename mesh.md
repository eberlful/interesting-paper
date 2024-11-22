1. Scaling Mesh Generation via Compressive Tokenization

## Scaling Mesh Generation via Compressive Tokenization

Github: https://github.com/whaohan/bpt

Paper: https://arxiv.org/abs/2411.07025

Date: 11.11.2024

##### Abstract
We propose a compressive yet effective mesh representation, Blocked and Patchified Tokenization (BPT), facilitating the generation of meshes exceeding 8k faces. BPT compresses mesh sequences by employing block-wise indexing and patch aggregation, reducing their length by approximately 75\% compared to the original sequences. This compression milestone unlocks the potential to utilize mesh data with significantly more faces, thereby enhancing detail richness and improving generation robustness. Empowered with the BPT, we have built a foundation mesh generative model training on scaled mesh data to support flexible control for point clouds and images. Our model demonstrates the capability to generate meshes with intricate details and accurate topology, achieving SoTA performance on mesh generation and reaching the level for direct product usage.
