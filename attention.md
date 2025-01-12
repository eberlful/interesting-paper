1. Star Attention: Efficient LLM Inference over Long Sequences
2. Attamba: Attending To Multi-Token States
3. KV Shifting Attention Enhances Language Modeling
4. Entropy-Guided Attention for Private LLMs


## Star Attention: Efficient LLM Inference over Long Sequences

Paper: https://arxiv.org/pdf/2411.17116

Date: 26.11.2024

##### Abstract
Inference with Transformer-based Large Language Models (LLMs) on long sequences is both costly and slow due to the quadratic complexity of the self-attention mechanism. We introduce Star Attention, a two-phase block-sparse approximation that improves computational efficiency by sharding attention across multiple hosts while minimizing communication overhead. In the first phase, the context is processed using blockwise-local attention across hosts, in parallel. In the second phase, query and response tokens attend to all prior cached tokens through sequence-global attention. Star Attention integrates seamlessly with most Transformer-based LLMs trained with global attention, reducing memory requirements and inference time by up to 11x while preserving 95-100% of accuracy.

## Attamba: Attending To Multi-Token States

Github: https://github.com/abdelfattah-lab/attamba

Paper: https://arxiv.org/abs/2411.17685

Date: 26.11.2024

##### Abstract
When predicting the next token in a sequence, vanilla transformers compute attention over all previous tokens, resulting in quadratic scaling of compute with sequence length. State-space models compress the entire sequence of tokens into a fixed-dimensional representation to improve efficiency, while other architectures achieve sub-quadratic complexity via low-rank projections or sparse attention patterns over the sequence. In this paper, we introduce Attamba, a novel architecture that uses state-space models to compress chunks of tokens and applies attention on these compressed key-value representations. We find that replacing key and value projections in a transformer with SSMs can improve model quality and enable flexible token chunking, resulting in 24% improved perplexity with transformer of similar KV-Cache and attention footprint, and ~4 times smaller KV-Cache and Attention FLOPs for 5% perplexity trade-off. Attamba can perform attention on chunked-sequences of variable length, enabling a smooth transition between quadratic and linear scaling, offering adaptable efficiency gains.

## KV Shifting Attention Enhances Language Modeling

Paper: https://arxiv.org/pdf/2411.19574

Date: 29.11.2024

##### Abstract
The current large language models are mainly based on decode-only structure transformers, which have great in-context learning (ICL) capabilities. It is generally believed that the important foundation of its ICL capability is the induction heads mechanism, which requires at least two layers attention. In order to more efficiently implement the ability of the model's induction, we revisit the induction heads mechanism and proposed a KV shifting attention. We theoretically prove that the KV shifting attention reducing the model's requirements for the depth and width of the induction heads mechanism. Our experimental results demonstrate that KV shifting attention is beneficial to learning induction heads and language modeling, which lead to better performance or faster convergence from toy models to the pre-training models with more than 10 B parameters.

## Entropy-Guided Attention for Private LLMs

Github: https://github.com/Nandan91/entropy-guided-attention-llm

Paper: https://arxiv.org/abs/2501.03489

Date: 07.01.2025

##### Abstract
The pervasiveness of proprietary language models has raised critical privacy concerns, necessitating advancements in private inference (PI), where computations are performed directly on encrypted data without revealing users' sensitive information. While PI offers a promising solution, its practical deployment is hindered by substantial communication and latency overheads, primarily stemming from nonlinear operations. To address this, we introduce an information-theoretic framework to characterize the role of nonlinearities in decoder-only language models, laying a principled foundation for optimizing transformer-architectures tailored to the demands of PI. By leveraging Shannon's entropy as a quantitative measure, we uncover the previously unexplored dual significance of nonlinearities: beyond ensuring training stability, they are crucial for maintaining attention head diversity. Specifically, we find that their removal triggers two critical failure modes: {\em entropy collapse} in deeper layers that destabilizes training, and {\em entropic overload} in earlier layers that leads to under-utilization of Multi-Head Attention's (MHA) representational capacity. We propose an entropy-guided attention mechanism paired with a novel entropy regularization technique to mitigate entropic overload. Additionally, we explore PI-friendly alternatives to layer normalization for preventing entropy collapse and stabilizing the training of LLMs with reduced-nonlinearities. Our study bridges the gap between information theory and architectural design, establishing entropy dynamics as a principled guide for developing efficient PI architectures. The code and implementation are available at https://github.com/Nandan91/entropy-guided-attention-llm{entropy-guided-llm}.
