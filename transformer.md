1. The Super Weight in Large Language Models
2. What Matters in Transformers? Not All Attention is Needed
3. TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters
4. Differential Transformer
5. Byte Latent Transformer: Patches Scale Better Than Tokens
6. Large Concept Models: Language Modeling in a Sentence Representation Space
7. Transformer^2: Self-adaptive LLMs
8. The Curse of Depth in Large Language Models
9. You Do Not Fully Utilize Transformer's Representation Capacity
10. LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers
11. On Relation-Specific Neurons in Large Language Models


## The Super Weight in Large Language Models

Github: https://github.com/mengxiayu/LLMSuperWeight

Paper: https://arxiv.org/pdf/2411.07191

Date: 11.11.2024

##### Abstract
Recent works have shown a surprising result: a small fraction of Large Language Model (LLM) parameter outliers are disproportionately important to the quality of the model. LLMs contain billions of parameters, so these small fractions, such as 0.01%, translate to hundreds of thousands of parameters. In this work, we present an even more surprising finding: Pruning as few as a single parameter can destroy an LLM’s ability to generate text – increasing perplexity by 3 orders of magnitude and reducing zero-shot accuracy to guessing. We propose a data-free method for identifying such parameters, termed super weights, using a single forward pass through the model. We additionally find that these super weights induce correspondingly rare and large activation outliers, termed super activations. When preserved with high precision, super activations can improve simple round-to-nearest quantization to become competitive with state-of-the-art methods. For weight quantization, we similarly find that by preserving the super weight and clipping other weight outliers, round-to-nearest quantization can scale to much larger block sizes than previously considered. To facilitate further research into super weights, we provide an index of super weight coordinates for common, openly available LLMs.

## What Matters in Transformers? Not All Attention is Needed

Github: https://github.com/CASE-Lab-UMD/LLM-Drop

Paper: https://arxiv.org/abs/2406.15786

Date: 22.06.2024

##### Abstract
While scaling Transformer-based large language models (LLMs) has demonstrated promising performance across various tasks, it also introduces redundant architectures, posing efficiency challenges for real-world deployment. Despite some recognition of redundancy in LLMs, the variability of redundancy across different architectures in transformers, such as MLP and Attention layers, is under-explored. In this work, we investigate redundancy across different modules within Transformers, including Blocks, MLP, and Attention layers, using a similarity-based metric. Surprisingly, despite the critical role of attention layers in distinguishing transformers from other architectures, we found that a large portion of these layers exhibit excessively high similarity and can be pruned without degrading performance. For instance, Llama-2-70B achieved a 48.4\% speedup with only a 2.4\% performance drop by pruning half of the attention layers. Furthermore, by tracing model checkpoints throughout the training process, we observed that attention layer redundancy is inherent and consistent across training stages. Additionally, we further propose a method that jointly drops Attention and MLP layers, allowing us to more aggressively drop additional layers. For instance, when dropping 31 layers (Attention + MLP), Llama-2-13B still retains 90\% of the performance on the MMLU task. Our work provides valuable insights for future network architecture design. The code is released at: https://github.com/CASE-Lab-UMD/LLM-Drop.

## TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters

Github: https://github.com/Haiyang-W/TokenFormer

Paper: https://arxiv.org/abs/2410.23168

Date: 30.10.2024

##### Abstract
Transformers have become the predominant architecture in foundation models due to their excellent performance across various domains. However, the substantial cost of scaling these models remains a significant concern. This problem arises primarily from their dependence on a fixed number of parameters within linear projections. When architectural modifications (e.g., channel dimensions) are introduced, the entire model typically requires retraining from scratch. As model sizes continue growing, this strategy results in increasingly high computational costs and becomes unsustainable. To overcome this problem, we introduce TokenFormer, a natively scalable architecture that leverages the attention mechanism not only for computations among input tokens but also for interactions between tokens and model parameters, thereby enhancing architectural flexibility. By treating model parameters as tokens, we replace all the linear projections in Transformers with our token-parameter attention layer, where input tokens act as queries and model parameters as keys and values. This reformulation allows for progressive and efficient scaling without necessitating retraining from scratch. Our model scales from 124M to 1.4B parameters by incrementally adding new key-value parameter pairs, achieving performance comparable to Transformers trained from scratch while greatly reducing training costs. Code and models are available at \url{this https URL}.

## Differential Transformer

Github: https://github.com/microsoft/unilm/tree/master/Diff-Transformer?utm_source=catalyzex.com

Paper: https://arxiv.org/abs/2410.05258

Date 07.10.2024

##### Abstract
Transformer tends to overallocate attention to irrelevant context. In this work, we introduce Diff Transformer, which amplifies attention to the relevant context while canceling noise. Specifically, the differential attention mechanism calculates attention scores as the difference between two separate softmax attention maps. The subtraction cancels noise, promoting the emergence of sparse attention patterns. Experimental results on language modeling show that Diff Transformer outperforms Transformer in various settings of scaling up model size and training tokens. More intriguingly, it offers notable advantages in practical applications, such as long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and reduction of activation outliers. By being less distracted by irrelevant context, Diff Transformer can mitigate hallucination in question answering and text summarization. For in-context learning, Diff Transformer not only enhances accuracy but is also more robust to order permutation, which was considered as a chronic robustness issue. The results position Diff Transformer as a highly effective and promising architecture to advance large language models.

## Byte Latent Transformer: Patches Scale Better Than Tokens

Paper: https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/

Date: 12.12.2024

##### Abstract
We introduce the Byte Latent Transformer (BLT), a new byte-level LLM architecture that, for the first time, matches tokenization-based LLM performance at scale with significant improvements in inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve as the primary units of computation. Patches are segmented dynamically based on the entropy of the next byte, allocating more compute and model capacity where increased data complexity demands it. We present the first flop controlled scaling study of byte-level models up to 8B parameters with 4T training bytes. Our results demonstrate the feasibility of scaling models trained on raw bytes without a fixed-vocabulary. Both training and inference efficiency improve due to dynamically selecting long patches when data is predictable, along with qualitative improvements on reasoning and long tail generalization. Overall, for fixed inference costs, BLT shows significantly better scaling than tokenization-based models, by simultaneously growing both patch and model size.

## Large Concept Models: Language Modeling in a Sentence Representation Space

Paper: https://ai.meta.com/research/publications/large-concept-models-language-modeling-in-a-sentence-representation-space/

Date: 11.12.2024

##### Abstract
LLMs have revolutionized the field of artificial intelligence and have emerged as the de-facto tool for many tasks. The current established technology of LLMs is to process input and generate output at the token level. This is in sharp contrast to humans who operate at multiple levels of abstraction, well beyond single words, to analyze information and to generate creative content. In this paper, we present an attempt at an architecture which operates on an explicit higher-level semantic representation, which we name a “concept”. Concepts are language- and modality-agnostic and represent a higher level idea or action in a flow. Hence, we build a“Large Concept Model”. In this study, as proof of feasibility, we assume that a concept corresponds to a sentence, and use an existing sentence embedding space, SONAR, which supports up to 200 languages in both text and speech modalities. The Large Concept Model is trained to perform autoregressive sentence prediction in an embedding space. We explore multiple approaches, namely MSE regression, variants of diffusion-based generation, and models operating in a quantized SONAR space. These explorations are performed using 1.6B parameter models and training data in the order of 1.3T tokens. We then scale one architecture to a model size of 7B parameters and training data of about 7.7T tokens. We perform an experimental evaluation on several generative tasks, namely summarization and a new task of summary expansion. Finally, we show that our model exhibits impressive zero-shot generalization performance to many languages, outperforming existing LLMs of the same size. The training code of our models is freely available.

## Transformer^2: Self-adaptive LLMs

Github: https://github.com/SakanaAI/self-adaptive-llms

Paper: https://arxiv.org/abs/2501.06252

Date: 14.01.2025

##### Abstract
Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse tasks. We introduce \implname, a novel self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices. During inference, \implname employs a two-pass mechanism: first, a dispatch system identifies the task properties, and then task-specific "expert" vectors, trained using reinforcement learning, are dynamically mixed to obtain targeted behavior for the incoming prompt. Our method outperforms ubiquitous approaches such as LoRA, with fewer parameters and greater efficiency. \implname demonstrates versatility across different LLM architectures and modalities, including vision-language tasks. \implname represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of LLMs, paving the way for truly dynamic, self-organizing AI systems.


## The Curse of Depth in Large Language Models

Paper: https://arxiv.org/abs/2502.05795

Date: 09.02.2025

##### Abstract
In this paper, we introduce the Curse of Depth, a concept that highlights, explains, and addresses the recent observation in modern Large Language Models(LLMs) where nearly half of the layers are less effective than expected. We first confirm the wide existence of this phenomenon across the most popular families of LLMs such as Llama, Mistral, DeepSeek, and Qwen. Our analysis, theoretically and empirically, identifies that the underlying reason for the ineffectiveness of deep layers in LLMs is the widespread usage of Pre-Layer Normalization (Pre-LN). While Pre-LN stabilizes the training of Transformer LLMs, its output variance exponentially grows with the model depth, which undesirably causes the derivative of the deep Transformer blocks to be an identity matrix, and therefore barely contributes to the training. To resolve this training pitfall, we propose LayerNorm Scaling, which scales the variance of output of the layer normalization inversely by the square root of its depth. This simple modification mitigates the output variance explosion of deeper Transformer layers, improving their contribution. Our experimental results, spanning model sizes from 130M to 1B, demonstrate that LayerNorm Scaling significantly enhances LLM pre-training performance compared to Pre-LN. Moreover, this improvement seamlessly carries over to supervised fine-tuning. All these gains can be attributed to the fact that LayerNorm Scaling enables deeper layers to contribute more effectively during training.


## You Do Not Fully Utilize Transformer's Representation Capacity

Paper: https://arxiv.org/abs/2502.09245

Date: 13.02.2025

##### Abstract
In contrast to RNNs, which compress previous tokens into a single hidden state, Transformers can attend to all previous tokens directly. However, standard Transformers only use representations from the immediately preceding layer. In this paper, we show that this design choice causes representation collapse and leads to suboptimal performance. To address this issue, we introduce Layer-Integrated Memory (LIMe), a simple yet powerful approach that preserves the model's overall memory footprint while expanding its representational capacity by allowing access to hidden states from earlier layers. Through extensive experiments across various architectures and different lookup mechanisms, we demonstrate consistent performance improvements on a wide range of tasks. Moreover, our analysis of the learned representation dynamics and our exploration of depthwise circuits reveal how LIMe integrates information across layers, pointing to promising directions for future research.

## LLM-Microscope: Uncovering the Hidden Role of Punctuation in Context Memory of Transformers

Paper: https://arxiv.org/abs/2502.15007

Date: 20.02.2025

##### Abstract
We introduce methods to quantify how Large Language Models (LLMs) encode and store contextual information, revealing that tokens often seen as minor (e.g., determiners, punctuation) carry surprisingly high context. Notably, removing these tokens -- especially stopwords, articles, and commas -- consistently degrades performance on MMLU and BABILong-4k, even if removing only irrelevant tokens. Our analysis also shows a strong correlation between contextualization and linearity, where linearity measures how closely the transformation from one layer's embeddings to the next can be approximated by a single linear mapping. These findings underscore the hidden importance of filler tokens in maintaining context. For further exploration, we present LLM-Microscope, an open-source toolkit that assesses token-level nonlinearity, evaluates contextual memory, visualizes intermediate layer contributions (via an adapted Logit Lens), and measures the intrinsic dimensionality of representations. This toolkit illuminates how seemingly trivial tokens can be critical for long-range understanding.

## On Relation-Specific Neurons in Large Language Models

Github: https://github.com/cisnlp/relation-specific-neurons

Paper: https://arxiv.org/abs/2502.17355

Date: 24.02.2025

##### Abstract
In large language models (LLMs), certain neurons can store distinct pieces of knowledge learned during pretraining. While knowledge typically appears as a combination of relations and entities, it remains unclear whether some neurons focus on a relation itself -- independent of any entity. We hypothesize such neurons detect a relation in the input text and guide generation involving such a relation. To investigate this, we study the Llama-2 family on a chosen set of relations with a statistics-based method. Our experiments demonstrate the existence of relation-specific neurons. We measure the effect of selectively deactivating candidate neurons specific to relation r on the LLM's ability to handle (1) facts whose relation is r and (2) facts whose relation is a different relation r' neq r. With respect to their capacity for encoding relation information, we give evidence for the following three properties of relation-specific neurons. (i) Neuron cumulativity. The neurons for r present a cumulative effect so that deactivating a larger portion of them results in the degradation of more facts in r. (ii) Neuron versatility. Neurons can be shared across multiple closely related as well as less related relations. Some relation neurons transfer across languages. (iii) Neuron interference. Deactivating neurons specific to one relation can improve LLM generation performance for facts of other relations. We will make our code publicly available at https://github.com/cisnlp/relation-specific-neurons.
