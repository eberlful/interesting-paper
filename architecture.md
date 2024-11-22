1. Hymba: A Hybrid-head Architecture for Small Language Models
2. Natural Language Reinforcement Learning
3. nGPT: Normalized Transformer with Representation Learning on the Hypersphere
4. Differential Transformer
5. What Matters in Transformers? Not All Attention is Needed


## Hymba: A Hybrid-head Architecture for Small Language Models

Github: https://github.com/NVlabs/hymba

Paper: https://arxiv.org/pdf/2411.13676

Date: 20.11.2024

##### Abstract
We propose Hymba, a family of small language models featuring a hybrid-head parallel architecture that integrates transformer attention mechanisms with state space models (SSMs) for enhanced efficiency. Attention heads provide high-resolution recall, while SSM heads enable efficient context summarization. Additionally, we introduce learnable meta tokens that are prepended to prompts, storing critical information and alleviating the "forced-to-attend" burden associated with attention mechanisms. This model is further optimized by incorporating cross-layer key-value (KV) sharing and partial sliding window attention, resulting in a compact cache size. During development, we conducted a controlled study comparing various architectures under identical settings and observed significant advantages of our proposed architecture. Notably, Hymba achieves state-of-the-art results for small LMs: Our Hymba-1.5B-Base model surpasses all sub-2B public models in performance and even outperforms Llama-3.2-3B with 1.32% higher average accuracy, an 11.67x cache size reduction, and 3.49x throughput.

## Natural Language Reinforcement Learning

Github: https://github.com/waterhorse1/Natural-language-RL

Paper: https://arxiv.org/pdf/2411.14251

Date: 21.11.2024

##### Abstract
Reinforcement Learning (RL) mathematically formulates decision-making with Markov Decision Process (MDP). With MDPs, researchers have achieved remarkable breakthroughs across various domains, including games, robotics, and language models. This paper seeks a new possibility, Natural Language Reinforcement Learning (NLRL), by extending traditional MDP to natural language-based representation space. Specifically, NLRL innovatively redefines RL principles, including task objectives, policy, value function, Bellman equation, and policy iteration, into their language counterparts. With recent advancements in large language models (LLMs), NLRL can be practically implemented to achieve RL-like policy and value improvement by either pure prompting or gradient-based training. Experiments over Maze, Breakthrough, and Tic-Tac-Toe games demonstrate the effectiveness, efficiency, and interpretability of the NLRL framework among diverse use cases. Our code will be released at https://github.com/waterhorse1/Natural-language-RL.

## nGPT: Normalized Transformer with Representation Learning on the Hypersphere

Github: https://github.com/NVIDIA/ngpt

Paper: https://arxiv.org/abs/2410.01131

Date: 01.10.2024

##### Abstract
We propose a novel neural network architecture, the normalized Transformer (nGPT) with representation learning on the hypersphere. In nGPT, all vectors forming the embeddings, MLP, attention matrices and hidden states are unit norm normalized. The input stream of tokens travels on the surface of a hypersphere, with each layer contributing a displacement towards the target output predictions. These displacements are defined by the MLP and attention blocks, whose vector components also reside on the same hypersphere. Experiments show that nGPT learns much faster, reducing the number of training steps required to achieve the same accuracy by a factor of 4 to 20, depending on the sequence length.

## Differential Transformer

Github: https://github.com/microsoft/unilm/tree/master/Diff-Transformer?utm_source=catalyzex.com

Paper: https://arxiv.org/abs/2410.05258

Date: 07.10.2024

##### Abstract
Transformer tends to overallocate attention to irrelevant context. In this work, we introduce Diff Transformer, which amplifies attention to the relevant context while canceling noise. Specifically, the differential attention mechanism calculates attention scores as the difference between two separate softmax attention maps. The subtraction cancels noise, promoting the emergence of sparse attention patterns. Experimental results on language modeling show that Diff Transformer outperforms Transformer in various settings of scaling up model size and training tokens. More intriguingly, it offers notable advantages in practical applications, such as long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and reduction of activation outliers. By being less distracted by irrelevant context, Diff Transformer can mitigate hallucination in question answering and text summarization. For in-context learning, Diff Transformer not only enhances accuracy but is also more robust to order permutation, which was considered as a chronic robustness issue. The results position Diff Transformer as a highly effective and promising architecture to advance large language models.

## What Matters in Transformers? Not All Attention is Needed

Github: https://github.com/CASE-Lab-UMD/LLM-Drop

Paper: https://arxiv.org/abs/2406.15786

Date: 22.06.2024

##### Abstract
While scaling Transformer-based large language models (LLMs) has demonstrated promising performance across various tasks, it also introduces redundant architectures, posing efficiency challenges for real-world deployment. Despite some recognition of redundancy in LLMs, the variability of redundancy across different architectures in transformers, such as MLP and Attention layers, is under-explored. In this work, we investigate redundancy across different modules within Transformers, including Blocks, MLP, and Attention layers, using a similarity-based metric. Surprisingly, despite the critical role of attention layers in distinguishing transformers from other architectures, we found that a large portion of these layers exhibit excessively high similarity and can be pruned without degrading performance. For instance, Llama-2-70B achieved a 48.4\% speedup with only a 2.4\% performance drop by pruning half of the attention layers. Furthermore, by tracing model checkpoints throughout the training process, we observed that attention layer redundancy is inherent and consistent across training stages. Additionally, we further propose a method that jointly drops Attention and MLP layers, allowing us to more aggressively drop additional layers. For instance, when dropping 31 layers (Attention + MLP), Llama-2-13B still retains 90\% of the performance on the MMLU task. Our work provides valuable insights for future network architecture design. The code is released at: https://github.com/CASE-Lab-UMD/LLM-Drop.
