1. Hymba: A Hybrid-head Architecture for Small Language Models
2. Natural Language Reinforcement Learning


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
