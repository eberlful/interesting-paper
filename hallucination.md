1. Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models
2. Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States
3. DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations
4. Exploring Concept Depth: How Large Language Models Acquire Knowledge at Different Layers?
5. Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach
6. Do LLMs "know" internally when they follow instructions?
7. Do LLMs estimate uncertainty well in instruction-following?
8. Linear Correlation in LM's Compositional Generalization and Hallucination
9. The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering
10. When an LLM is apprehensive about its answers -- and when its uncertainty is justified
11. How to Steer LLM Latents for Hallucination Detection?
12. LettuceDetect: A Hallucination Detection Framework for RAG Applications


## Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models

Paper: https://arxiv.org/pdf/2411.14257

Date: 22.11.2024

##### Abstract
Hallucinations in large language models are a widespread problem, yet the mechanisms behind whether models will hallucinate are poorly understood, limiting our ability to solve this problem. Using sparse autoencoders as an interpretability tool, we discover that a key part of these mechanisms is entity recognition, where the model detects if an entity is one it can recall facts about. Sparse autoencoders uncover meaningful directions in the representation space, these detect whether the model recognizes an entity, e.g. detecting it doesn't know about an athlete or a movie. This suggests that models can have self-knowledge: internal representations about their own capabilities. These directions are causally relevant: capable of steering the model to refuse to answer questions about known entities, or to hallucinate attributes of unknown entities when it would otherwise refuse. We demonstrate that despite the sparse autoencoders being trained on the base model, these directions have a causal effect on the chat model's refusal behavior, suggesting that chat finetuning has repurposed this existing mechanism. Furthermore, we provide an initial exploration into the mechanistic role of these directions in the model, finding that they disrupt the attention of downstream heads that typically move entity attributes to the final token.

## Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States

Paper: https://arxiv.org/pdf/2402.09733

Date: 15.02.2024

##### Abstract
Large Language Models (LLMs) can make up answers that are not real, and this is known as hallucination. This research aims to see if, how, and to what extent LLMs are aware of hallucination. More specifically, we check whether and how an LLM reacts differently in its hidden states when it answers a question right versus when it hallucinates. To do this, we introduce an experimental framework which allows examining LLM's hidden states in different hallucination situations. Building upon this framework, we conduct a series of experiments with language models in the LLaMA family (Touvron et al., 2023). Our empirical findings suggest that LLMs react differently when processing a genuine response versus a fabricated one. We then apply various model interpretation techniques to help understand and explain the findings better. Moreover, informed by the empirical observations, we show great potential of using the guidance derived from LLM's hidden representation space to mitigate hallucination. We believe this work provides insights into how LLMs produce hallucinated answers and how to make them occur less often.

## DebUnc: Mitigating Hallucinations in Large Language Model Agent Communication with Uncertainty Estimations

Github: https://github.com/lukeyoffe/debunc

Paper: https://arxiv.org/pdf/2407.06426

Date: 08.07.2024

##### Abstract
To enhance Large Language Model (LLM) capabilities, multi-agent debates have been introduced, where multiple LLMs discuss solutions to a problem over several rounds of debate. However, LLMs often produce incorrect responses that appear deceptively confident, which can mislead other agents. This is partly because agents do not express their confidence levels during standard debates. To address this, we introduce DebUnc, a multi-agent debate framework that uses uncertainty metrics to assess agent confidence levels. We adapted the LLM attention mechanism to adjust token weights based on confidence levels and also explored using textual prompts to convey confidence. Our evaluations across various benchmarks show that attention-based methods are particularly effective, and that as uncertainty metrics evolve, performance will continue to increase. The code is available at this https URL

## Exploring Concept Depth: How Large Language Models Acquire Knowledge at Different Layers?

Github: https://github.com/Luckfort/CD

Paper: https://arxiv.org/pdf/2404.07066

Date: 10.04.2024

##### Abstract
Large language models (LLMs) have shown remarkable performances across a wide range of tasks. However, the mechanisms by which these models encode tasks of varying complexities remain poorly understood. In this paper, we explore the hypothesis that LLMs process concepts of varying complexities in different layers, introducing the idea of ``Concept Depth'' to suggest that more complex concepts are typically acquired in deeper layers. Specifically, we categorize concepts based on their level of abstraction, defining them in the order of increasing complexity within factual, emotional, and inferential tasks. We conduct extensive probing experiments using layer-wise representations across various LLM families (Gemma, LLaMA, Qwen) on various datasets spanning the three domains of tasks. Our findings reveal that models could efficiently conduct probing for simpler tasks in shallow layers, and more complex tasks typically necessitate deeper layers for accurate understanding. Additionally, we examine how external factors, such as adding noise to the input and quantizing the model weights, might affect layer-wise representations. Our findings suggest that these factors can impede the development of a conceptual understanding of LLMs until deeper layers are explored. We hope that our proposed concept and experimental insights will enhance the understanding of the mechanisms underlying LLMs. Our codes are available at \url{this https URL}.

## Uncertainty Estimation and Quantification for LLMs: A Simple Supervised Approach

Paper: https://arxiv.org/pdf/2404.15993

Date: 24.04.2024

##### Abstract
In this paper, we study the problem of uncertainty estimation and calibration for LLMs. We begin by formulating the uncertainty estimation problem, a relevant yet underexplored area in existing literature. We then propose a supervised approach that leverages labeled datasets to estimate the uncertainty in LLMs' responses. Based on the formulation, we illustrate the difference between the uncertainty estimation for LLMs and that for standard ML models and explain why the hidden neurons of the LLMs may contain uncertainty information. Our designed approach demonstrates the benefits of utilizing hidden activations to enhance uncertainty estimation across various tasks and shows robust transferability in out-of-distribution settings. We distinguish the uncertainty estimation task from the uncertainty calibration task and show that better uncertainty estimation leads to better calibration performance. Furthermore, our method is easy to implement and adaptable to different levels of model accessibility including black box, grey box, and white box.

## Do LLMs "know" internally when they follow instructions?

Paper: https://arxiv.org/pdf/2410.14516

Date: 18.10.2024

##### Abstract
Instruction-following is crucial for building AI agents with large language models (LLMs), as these models must adhere strictly to user-provided constraints and guidelines. However, LLMs often fail to follow even simple and clear instructions. To improve instruction-following behavior and prevent undesirable outputs, a deeper understanding of how LLMs' internal states relate to these outcomes is required. Our analysis of LLM internal states reveal a dimension in the input embedding space linked to successful instruction-following. We demonstrate that modifying representations along this dimension improves instruction-following success rates compared to random changes, without compromising response quality. Further investigation reveals that this dimension is more closely related to the phrasing of prompts rather than the inherent difficulty of the task or instructions. This discovery also suggests explanations for why LLMs sometimes fail to follow clear instructions and why prompt engineering is often effective, even when the content remains largely unchanged. This work provides insight into the internal workings of LLMs' instruction-following, paving the way for reliable LLM agents.

## Do LLMs estimate uncertainty well in instruction-following?

Paper: https://arxiv.org/pdf/2410.14582

Date: 18.10.2024

##### Abstract
Large language models (LLMs) could be valuable personal AI agents across various domains, provided they can precisely follow user instructions. However, recent studies have shown significant limitations in LLMs' instruction-following capabilities, raising concerns about their reliability in high-stakes applications. Accurately estimating LLMs' uncertainty in adhering to instructions is critical to mitigating deployment risks. We present, to our knowledge, the first systematic evaluation of the uncertainty estimation abilities of LLMs in the context of instruction-following. Our study identifies key challenges with existing instruction-following benchmarks, where multiple factors are entangled with uncertainty stems from instruction-following, complicating the isolation and comparison across methods and models. To address these issues, we introduce a controlled evaluation setup with two benchmark versions of data, enabling a comprehensive comparison of uncertainty estimation methods under various conditions. Our findings show that existing uncertainty methods struggle, particularly when models make subtle errors in instruction following. While internal model states provide some improvement, they remain inadequate in more complex scenarios. The insights from our controlled evaluation setups provide a crucial understanding of LLMs' limitations and potential for uncertainty estimation in instruction-following tasks, paving the way for more trustworthy AI agents.

## Linear Correlation in LM's Compositional Generalization and Hallucination

Github: https://github.com/KomeijiForce/LinCorr

Paper: https://arxiv.org/abs/2502.04520

Date: 06.02.2025

##### Abstract
The generalization of language models (LMs) is undergoing active debates, contrasting their potential for general intelligence with their struggles with basic knowledge composition (e.g., reverse/transition curse). This paper uncovers the phenomenon of linear correlations in LMs during knowledge composition. For explanation, there exists a linear transformation between certain related knowledge that maps the next token prediction logits from one prompt to another, e.g., "X lives in the city of" rightarrow "X lives in the country of" for every given X. This mirrors the linearity in human knowledge composition, such as Paris rightarrow France. Our findings indicate that the linear transformation is resilient to large-scale fine-tuning, generalizing updated knowledge when aligned with real-world relationships, but causing hallucinations when it deviates. Empirical results suggest that linear correlation can serve as a potential identifier of LM's generalization. Finally, we show such linear correlations can be learned with a single feedforward network and pre-trained vocabulary representations, indicating LM generalization heavily relies on the latter.

## The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering

Paper: https://arxiv.org/abs/2502.03628

Date: 05.02.2025

##### Abstract
Large Vision-Language Models (LVLMs) can reason effectively over both textual and visual inputs, but they tend to hallucinate syntactically coherent yet visually ungrounded contents. In this paper, we investigate the internal dynamics of hallucination by examining the tokens logits rankings throughout the generation process, revealing three key patterns in how LVLMs process information: (1) gradual visual information loss -- visually grounded tokens gradually become less favored throughout generation, and (2) early excitation -- semantically meaningful tokens achieve peak activation in the layers earlier than the final layer. (3) hidden genuine information -- visually grounded tokens though not being eventually decided still retain relatively high rankings at inference. Based on these insights, we propose VISTA (Visual Information Steering with Token-logit Augmentation), a training-free inference-time intervention framework that reduces hallucination while promoting genuine information. VISTA works by combining two complementary approaches: reinforcing visual information in activation space and leveraging early layer activations to promote semantically meaningful decoding. Compared to existing methods, VISTA requires no external supervision and is applicable to various decoding strategies. Extensive experiments show that VISTA on average reduces hallucination by abount 40% on evaluated open-ended generation task, and it consistently outperforms existing methods on four benchmarks across four architectures under three decoding strategies.

## When an LLM is apprehensive about its answers -- and when its uncertainty is justified

Github: https://github.com/LabARSS/question-complextiy-estimation

Paper: https://arxiv.org/abs/2503.01688

Date: 03.03.2025

##### Abstract
Uncertainty estimation is crucial for evaluating Large Language Models (LLMs), particularly in high-stakes domains where incorrect answers result in significant consequences. Numerous approaches consider this problem, while focusing on a specific type of uncertainty, ignoring others. We investigate what estimates, specifically token-wise entropy and model-as-judge (MASJ), would work for multiple-choice question-answering tasks for different question topics. Our experiments consider three LLMs: Phi-4, Mistral, and Qwen of different sizes from 1.5B to 72B and 14 topics. While MASJ performs similarly to a random error predictor, the response entropy predicts model error in knowledge-dependent domains and serves as an effective indicator of question difficulty: for biology ROC AUC is 0.73. This correlation vanishes for the reasoning-dependent domain: for math questions ROC-AUC is 0.55. More principally, we found out that the entropy measure required a reasoning amount. Thus, data-uncertainty related entropy should be integrated within uncertainty estimates frameworks, while MASJ requires refinement. Moreover, existing MMLU-Pro samples are biased, and should balance required amount of reasoning for different subdomains to provide a more fair assessment of LLMs performance.

## How to Steer LLM Latents for Hallucination Detection?

Paper: https://arxiv.org/abs/2503.01917

Date: 01.03.2025

##### Abstract
Hallucinations in LLMs pose a significant concern to their safe deployment in real-world applications. Recent approaches have leveraged the latent space of LLMs for hallucination detection, but their embeddings, optimized for linguistic coherence rather than factual accuracy, often fail to clearly separate truthful and hallucinated content. To this end, we propose the Truthfulness Separator Vector (TSV), a lightweight and flexible steering vector that reshapes the LLM's representation space during inference to enhance the separation between truthful and hallucinated outputs, without altering model parameters. Our two-stage framework first trains TSV on a small set of labeled exemplars to form compact and well-separated clusters. It then augments the exemplar set with unlabeled LLM generations, employing an optimal transport-based algorithm for pseudo-labeling combined with a confidence-based filtering process. Extensive experiments demonstrate that TSV achieves state-of-the-art performance with minimal labeled data, exhibiting strong generalization across datasets and providing a practical solution for real-world LLM applications.

## LettuceDetect: A Hallucination Detection Framework for RAG Applications

Github: https://github.com/KRLabsOrg/LettuceDetect

Paper: https://arxiv.org/abs/2502.17125

Date: 24.02.2025

##### Abstract
Retrieval Augmented Generation (RAG) systems remain vulnerable to hallucinated answers despite incorporating external knowledge sources. We present LettuceDetect a framework that addresses two critical limitations in existing hallucination detection methods: (1) the context window constraints of traditional encoder-based methods, and (2) the computational inefficiency of LLM based approaches. Building on ModernBERT's extended context capabilities (up to 8k tokens) and trained on the RAGTruth benchmark dataset, our approach outperforms all previous encoder-based models and most prompt-based models, while being approximately 30 times smaller than the best models. LettuceDetect is a token-classification model that processes context-question-answer triples, allowing for the identification of unsupported claims at the token level. Evaluations on the RAGTruth corpus demonstrate an F1 score of 79.22% for example-level detection, which is a 14.8% improvement over Luna, the previous state-of-the-art encoder-based architecture. Additionally, the system can process 30 to 60 examples per second on a single GPU, making it more practical for real-world RAG applications.
