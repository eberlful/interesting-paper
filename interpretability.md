1. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet
2. I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders
3. Truth Neurons


## Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet

Paper: 
* https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
* https://www.anthropic.com/news/mapping-mind-language-model

Date: 21.05.2024

Golden Gate Bridge

## I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders

Github: https://github.com/AIRI-Institute/SAE-Reasoning

Paper: https://arxiv.org/abs/2503.18878

Date: 24.03.2025

##### Abstract
Large Language Models (LLMs) have achieved remarkable success in natural language processing. Recent advances have led to the developing of a new class of reasoning LLMs; for example, open-source DeepSeek-R1 has achieved state-of-the-art performance by integrating deep thinking and complex reasoning. Despite these impressive capabilities, the internal reasoning mechanisms of such models remain unexplored. In this work, we employ Sparse Autoencoders (SAEs), a method to learn a sparse decomposition of latent representations of a neural network into interpretable features, to identify features that drive reasoning in the DeepSeek-R1 series of models. First, we propose an approach to extract candidate ''reasoning features'' from SAE representations. We validate these features through empirical analysis and interpretability methods, demonstrating their direct correlation with the model's reasoning abilities. Crucially, we demonstrate that steering these features systematically enhances reasoning performance, offering the first mechanistic account of reasoning in LLMs. Code available at https://github.com/AIRI-Institute/SAE-Reasoning

## Truth Neurons

Paper: https://arxiv.org/abs/2505.12182

Date: 18.05.2025

##### Abstract
Despite their remarkable success and deployment across diverse workflows, language models sometimes produce untruthful responses. Our limited understanding of how truthfulness is mechanistically encoded within these models jeopardizes their reliability and safety. In this paper, we propose a method for identifying representations of truthfulness at the neuron level. We show that language models contain truth neurons, which encode truthfulness in a subject-agnostic manner. Experiments conducted across models of varying scales validate the existence of truth neurons, confirming that the encoding of truthfulness at the neuron level is a property shared by many language models. The distribution patterns of truth neurons over layers align with prior findings on the geometry of truthfulness. Selectively suppressing the activations of truth neurons found through the TruthfulQA dataset degrades performance both on TruthfulQA and on other benchmarks, showing that the truthfulness mechanisms are not tied to a specific dataset. Our results offer novel insights into the mechanisms underlying truthfulness in language models and highlight potential directions toward improving their trustworthiness and reliability.
