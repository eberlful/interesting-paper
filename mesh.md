1. Scaling Mesh Generation via Compressive Tokenization
2. Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation
3. TEXGen: a Generative Diffusion Model for Mesh Textures
4. SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE
5. Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale
6. SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images
7. CaPa: Carve-n-Paint Synthesis for Efficient 4K Textured Mesh Generation

## Scaling Mesh Generation via Compressive Tokenization

Github: https://github.com/whaohan/bpt

Paper: https://arxiv.org/abs/2411.07025

Date: 11.11.2024

##### Abstract
We propose a compressive yet effective mesh representation, Blocked and Patchified Tokenization (BPT), facilitating the generation of meshes exceeding 8k faces. BPT compresses mesh sequences by employing block-wise indexing and patch aggregation, reducing their length by approximately 75\% compared to the original sequences. This compression milestone unlocks the potential to utilize mesh data with significantly more faces, thereby enhancing detail richness and improving generation robustness. Empowered with the BPT, we have built a foundation mesh generative model training on scaled mesh data to support flexible control for point clouds and images. Our model demonstrates the capability to generate meshes with intricate details and accurate topology, achieving SoTA performance on mesh generation and reaching the level for direct product usage.

## Baking Gaussian Splatting into Diffusion Denoiser for Fast and Scalable Single-stage Image-to-3D Generation

Github: https://caiyuanhao1998.github.io/project/DiffusionGS/

Paper: https://arxiv.org/pdf/2411.14384

Date: 21.11.2024

##### Abstract
Existing feed-forward image-to-3D methods mainly rely on 2D multi-view diffusion models that cannot guarantee 3D consistency. These methods easily collapse when changing the prompt view direction and mainly handle object-centric prompt images. In this paper, we propose a novel single-stage 3D diffusion model, DiffusionGS, for object and scene generation from a single view. DiffusionGS directly outputs 3D Gaussian point clouds at each timestep to enforce view consistency and allow the model to generate robustly given prompt views of any directions, beyond object-centric inputs. Plus, to improve the capability and generalization ability of DiffusionGS, we scale up 3D training data by developing a scene-object mixed training strategy. Experiments show that our method enjoys better generation quality (2.20 dB higher in PSNR and 23.25 lower in FID) and over 5x faster speed (~6s on an A100 GPU) than SOTA methods. The user study and text-to-3D applications also reveals the practical values of our method. Our Project page at https://caiyuanhao1998.github.io/project/DiffusionGS/ shows the video and interactive generation results.

## TEXGen: a Generative Diffusion Model for Mesh Textures

Github: https://cvmi-lab.github.io/TEXGen/

Paper: https://arxiv.org/pdf/2411.14740

Date: 22.11.2024

## Abstract
While high-quality texture maps are essential for realistic 3D asset rendering, few studies have explored learning directly in the texture space, especially on large-scale datasets. In this work, we depart from the conventional approach of relying on pre-trained 2D diffusion models for test-time optimization of 3D textures. Instead, we focus on the fundamental problem of learning in the UV texture space itself. For the first time, we train a large diffusion model capable of directly generating high-resolution texture maps in a feed-forward manner. To facilitate efficient learning in high-resolution UV spaces, we propose a scalable network architecture that interleaves convolutions on UV maps with attention layers on point clouds. Leveraging this architectural design, we train a 700 million parameter diffusion model that can generate UV texture maps guided by text prompts and single-view images. Once trained, our model naturally supports various extended applications, including text-guided texture inpainting, sparse-view texture completion, and text-driven texture synthesis. Project page is at http://cvmi-lab.github.io/TEXGen/.

## SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE

Github: https://github.com/cyw-3d/SAR3D

Paper: https://arxiv.org/pdf/2411.16856

Date: 25.11.2024

##### Abstract
Autoregressive models have demonstrated remarkable success across various fields, from large language models (LLMs) to large multimodal models (LMMs) and 2D content generation, moving closer to artificial general intelligence (AGI). Despite these advances, applying autoregressive approaches to 3D object generation and understanding remains largely unexplored. This paper introduces Scale AutoRegressive 3D (SAR3D), a novel framework that leverages a multi-scale 3D vector-quantized variational autoencoder (VQVAE) to tokenize 3D objects for efficient autoregressive generation and detailed understanding. By predicting the next scale in a multi-scale latent representation instead of the next single token, SAR3D reduces generation time significantly, achieving fast 3D object generation in just 0.82 seconds on an A6000 GPU. Additionally, given the tokens enriched with hierarchical 3D-aware information, we finetune a pretrained LLM on them, enabling multimodal comprehension of 3D content. Our experiments show that SAR3D surpasses current 3D generation methods in both speed and quality and allows LLMs to interpret and caption 3D models comprehensively.


## Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale

Paper: https://arxiv.org/abs/2412.09548

Date: 12.12.2024

##### Abstract
Meshes are fundamental representations of 3D surfaces. However, creating high-quality meshes is a labor-intensive task that requires significant time and expertise in 3D modeling. While a delicate object often requires over 104 faces to be accurately modeled, recent attempts at generating artist-like meshes are limited to 1.6K faces and heavy discretization of vertex coordinates. Hence, scaling both the maximum face count and vertex coordinate resolution is crucial to producing high-quality meshes of realistic, complex 3D objects. We present Meshtron, a novel autoregressive mesh generation model able to generate meshes with up to 64K faces at 1024-level coordinate resolution --over an order of magnitude higher face count and 8× higher coordinate resolution than current state-of-the-art methods. Meshtron's scalability is driven by four key components: (1) an hourglass neural architecture, (2) truncated sequence training, (3) sliding window inference, (4) a robust sampling strategy that enforces the order of mesh sequences. This results in over 50% less training memory, 2.5× faster throughput, and better consistency than existing works. Meshtron generates meshes of detailed, complex 3D objects at unprecedented levels of resolution and fidelity, closely resembling those created by professional artists, and opening the door to more realistic generation of detailed 3D assets for animation, gaming, and virtual environments.

## SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images

Github: https://spar3d.github.io/

Paper: https://arxiv.org/abs/2501.04689

Date: 08.01.2025

##### Abstract
We study the problem of single-image 3D object reconstruction. Recent works have diverged into two directions: regression-based modeling and generative modeling. Regression methods efficiently infer visible surfaces, but struggle with occluded regions. Generative methods handle uncertain regions better by modeling distributions, but are computationally expensive and the generation is often misaligned with visible surfaces. In this paper, we present SPAR3D, a novel two-stage approach aiming to take the best of both directions. The first stage of SPAR3D generates sparse 3D point clouds using a lightweight point diffusion model, which has a fast sampling speed. The second stage uses both the sampled point cloud and the input image to create highly detailed meshes. Our two-stage design enables probabilistic modeling of the ill-posed single-image 3D task while maintaining high computational efficiency and great output fidelity. Using point clouds as an intermediate representation further allows for interactive user edits. Evaluated on diverse datasets, SPAR3D demonstrates superior performance over previous state-of-the-art methods, at an inference speed of 0.7 seconds. Project page with code and model: https://spar3d.github.io

## CaPa: Carve-n-Paint Synthesis for Efficient 4K Textured Mesh Generation

Github: https://ncsoft.github.io/CaPa/

Paper: https://arxiv.org/abs/2501.09433

Date: 16.01.2025

##### Abstract
The synthesis of high-quality 3D assets from textual or visual inputs has become a central objective in modern generative modeling. Despite the proliferation of 3D generation algorithms, they frequently grapple with challenges such as multi-view inconsistency, slow generation times, low fidelity, and surface reconstruction problems. While some studies have addressed some of these issues, a comprehensive solution remains elusive. In this paper, we introduce CaPa, a carve-and-paint framework that generates high-fidelity 3D assets efficiently. CaPa employs a two-stage process, decoupling geometry generation from texture synthesis. Initially, a 3D latent diffusion model generates geometry guided by multi-view inputs, ensuring structural consistency across perspectives. Subsequently, leveraging a novel, model-agnostic Spatially Decoupled Attention, the framework synthesizes high-resolution textures (up to 4K) for a given geometry. Furthermore, we propose a 3D-aware occlusion inpainting algorithm that fills untextured regions, resulting in cohesive results across the entire model. This pipeline generates high-quality 3D assets in less than 30 seconds, providing ready-to-use outputs for commercial applications. Experimental results demonstrate that CaPa excels in both texture fidelity and geometric stability, establishing a new standard for practical, scalable 3D asset generation.
