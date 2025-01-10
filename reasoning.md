1. Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions
2. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models
3. The Surprising Effectiveness of Test-Time Training for Abstract Reasoning
4. LLMs Do Not Think Step-by-step In Implicit Reasoning
5. Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability
6. Training Large Language Models to Reason in a Continuous Latent Space
7. Deliberation in Latent Space via Differentiable Cache Augmentation
8. Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces
9. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs
10. Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search
11. Test-time Computing: from System-1 Thinking to System-2 Thinking


## Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions

Github: https://github.com/AIDC-AI/Marco-o1

Paper: https://arxiv.org/pdf/2411.14405

Date: 21.11.2024

##### Abstract
Currently OpenAI o1 has sparked a surge of interest in the study of large reasoning models (LRM). Building on this momentum, Marco-o1 not only focuses on disciplines with standard answers, such as mathematics, physics, and coding -- which are well-suited for reinforcement learning (RL) -- but also places greater emphasis on open-ended resolutions. We aim to address the question: "Can the o1 model effectively generalize to broader domains where clear standards are absent and rewards are challenging to quantify?" Marco-o1 is powered by Chain-of-Thought (CoT) fine-tuning, Monte Carlo Tree Search (MCTS), reflection mechanisms, and innovative reasoning strategies -- optimized for complex real-world problem-solving tasks.


## GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models

Paper: https://arxiv.org/pdf/2410.05229

Date: 07.10.2024

##### Abstract
Recent advancements in Large Language Models (LLMs) have sparked interest in their formal reasoning capabilities, particularly in mathematics. The GSM8K benchmark is widely used to assess the mathematical reasoning of models on grade-school-level questions. While the performance of LLMs on GSM8K has significantly improved in recent years, it remains unclear whether their mathematical reasoning capabilities have genuinely advanced, raising questions about the reliability of the reported metrics. To address these concerns, we conduct a large-scale study on several SOTA open and closed models. To overcome the limitations of existing evaluations, we introduce GSM-Symbolic, an improved benchmark created from symbolic templates that allow for the generation of a diverse set of questions. GSM-Symbolic enables more controllable evaluations, providing key insights and more reliable metrics for measuring the reasoning capabilities of this http URL findings reveal that LLMs exhibit noticeable variance when responding to different instantiations of the same question. Specifically, the performance of all models declines when only the numerical values in the question are altered in the GSM-Symbolic benchmark. Furthermore, we investigate the fragility of mathematical reasoning in these models and show that their performance significantly deteriorates as the number of clauses in a question increases. We hypothesize that this decline is because current LLMs cannot perform genuine logical reasoning; they replicate reasoning steps from their training data. Adding a single clause that seems relevant to the question causes significant performance drops (up to 65%) across all state-of-the-art models, even though the clause doesn't contribute to the reasoning chain needed for the final answer. Overall, our work offers a more nuanced understanding of LLMs' capabilities and limitations in mathematical reasoning.

## The Surprising Effectiveness of Test-Time Training for Abstract Reasoning

Github: https://github.com/ekinakyurek/marc

Paper: https://arxiv.org/pdf/2411.07279

Date: 11.11.2024

##### Abstract
Language models have shown impressive performance on tasks within their training distribution, but often struggle with novel problems requiring complex reasoning. We investigate the effectiveness of test-time training (TTT) -- updating model parameters temporarily during inference using a loss derived from input data -- as a mechanism for improving models' reasoning capabilities, using the Abstraction and Reasoning Corpus (ARC) as a benchmark. Through systematic experimentation, we identify three crucial components for successful TTT: (1) initial finetuning on similar tasks (2) auxiliary task format and augmentations (3) per-instance training. TTT significantly improves performance on ARC tasks, achieving up to 6x improvement in accuracy compared to base fine-tuned models; applying TTT to an 8B-parameter language model, we achieve 53% accuracy on the ARC's public validation set, improving the state-of-the-art by nearly 25% for public and purely neural approaches. By ensembling our method with recent program generation approaches, we get SoTA public validation accuracy of 61.9%, matching the average human score. Our findings suggest that explicit symbolic search is not the only path to improved abstract reasoning in neural language models; additional test-time applied to continued training on few-shot examples can also be extremely effective.

## LLMs Do Not Think Step-by-step In Implicit Reasoning

Paper: https://arxiv.org/pdf/2411.15862

Date: 24.11.2024

##### Abstract
It has been well-known that Chain-of-Thought can remarkably enhance LLMs' performance on complex tasks. However, because it also introduces slower inference speeds and higher computational costs, many researches have attempted to use implicit CoT, which does not need LLMs to explicitly generate the intermediate steps. But there is still gap between their efficacy and typical explicit CoT methods. This leaves us a doubt that, does implicit CoT really equal to explicit CoT? Therefore, in this study, we address this question through experiments. We probe the information of intermediate steps from the model's hidden states when it is performing implicit CoT. The results surprisingly indicate that LLMs hardly think about intermediate steps, suggesting they may just rely on experience rather than strict step-by-step reasoning. Moreover, we find LLMs' implicit reasoning capabilities are susceptible and unstable, reaffirming the necessity of explicit CoT to effectively support complex tasks.

## Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability

Paper: https://arxiv.org/abs/2411.19943

Date: 29.11.2024

##### Abstract
Large Language Models (LLMs) have exhibited remarkable performance on reasoning tasks. They utilize autoregressive token generation to construct reasoning trajectories, enabling the development of a coherent chain of thought. In this work, we explore the impact of individual tokens on the final outcomes of reasoning tasks. We identify the existence of ``critical tokens'' that lead to incorrect reasoning trajectories in LLMs. Specifically, we find that LLMs tend to produce positive outcomes when forced to decode other tokens instead of critical tokens. Motivated by this observation, we propose a novel approach - cDPO - designed to automatically recognize and conduct token-level rewards for the critical tokens during the alignment process. Specifically, we develop a contrastive estimation approach to automatically identify critical tokens. It is achieved by comparing the generation likelihood of positive and negative models. To achieve this, we separately fine-tune the positive and negative models on various reasoning trajectories, consequently, they are capable of identifying identify critical tokens within incorrect trajectories that contribute to erroneous outcomes. Moreover, to further align the model with the critical token information during the alignment process, we extend the conventional DPO algorithms to token-level DPO and utilize the differential likelihood from the aforementioned positive and negative model as important weight for token-level DPO learning.Experimental results on GSM8K and MATH500 benchmarks with two-widely used models Llama-3 (8B and 70B) and deepseek-math (7B) demonstrate the effectiveness of the propsoed approach cDPO.

## Training Large Language Models to Reason in a Continuous Latent Space

Paper: https://arxiv.org/abs/2412.06769

Date: 09.12.2024

##### Abstract
Large language models (LLMs) are restricted to reason in the "language space", where they typically express the reasoning process with a chain-of-thought (CoT) to solve a complex reasoning problem. However, we argue that language space may not always be optimal for reasoning. For example, most word tokens are primarily for textual coherence and not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs. To explore the potential of LLM reasoning in an unrestricted latent space instead of using natural language, we introduce a new paradigm Coconut (Chain of Continuous Thought). We utilize the last hidden state of the LLM as a representation of the reasoning state (termed "continuous thought"). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space. Experiments show that Coconut can effectively augment the LLM on several reasoning tasks. This novel latent reasoning paradigm leads to emergent advanced reasoning patterns: the continuous thought can encode multiple alternative next reasoning steps, allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing to a single deterministic path like CoT. Coconut outperforms CoT in certain logical reasoning tasks that require substantial backtracking during planning, with fewer thinking tokens during inference. These findings demonstrate the promise of latent reasoning and offer valuable insights for future research.

## Deliberation in Latent Space via Differentiable Cache Augmentation

Paper: https://arxiv.org/abs/2412.17747

Date: 23.12.2024

##### Abstract
Techniques enabling large language models (LLMs) to "think more" by generating and attending to intermediate reasoning steps have shown promise in solving complex problems. However, the standard approaches generate sequences of discrete tokens immediately before responding, and so they can incur significant latency costs and be challenging to optimize. In this work, we demonstrate that a frozen LLM can be augmented with an offline coprocessor that operates on the model's key-value (kv) cache. This coprocessor augments the cache with a set of latent embeddings designed to improve the fidelity of subsequent decoding. We train this coprocessor using the language modeling loss from the decoder on standard pretraining data, while keeping the decoder itself frozen. This approach enables the model to learn, in an end-to-end differentiable fashion, how to distill additional computation into its kv-cache. Because the decoder remains unchanged, the coprocessor can operate offline and asynchronously, and the language model can function normally if the coprocessor is unavailable or if a given cache is deemed not to require extra computation. We show experimentally that when a cache is augmented, the decoder achieves lower perplexity on numerous subsequent tokens. Furthermore, even without any task-specific training, our experiments demonstrate that cache augmentation consistently reduces perplexity and improves performance across a range of reasoning-intensive tasks.

## Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces

Github: https://github.com/vision-x-nyu/thinking-in-space

Paper: https://arxiv.org/abs/2412.14171

Date: 18.12.2024

##### Abstract
Humans possess the visual-spatial intelligence to remember spaces from sequential visual observations. However, can Multimodal Large Language Models (MLLMs) trained on million-scale video datasets also ``think in space'' from videos? We present a novel video-based visual-spatial intelligence benchmark (VSI-Bench) of over 5,000 question-answer pairs, and find that MLLMs exhibit competitive - though subhuman - visual-spatial intelligence. We probe models to express how they think in space both linguistically and visually and find that while spatial reasoning capabilities remain the primary bottleneck for MLLMs to reach higher benchmark performance, local world models and spatial awareness do emerge within these models. Notably, prevailing linguistic reasoning techniques (e.g., chain-of-thought, self-consistency, tree-of-thoughts) fail to improve performance, whereas explicitly generating cognitive maps during question-answering enhances MLLMs' spatial distance ability.

## Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs

Paper: https://arxiv.org/abs/2412.21187

Date: 30.12.2024

##### Abstract
The remarkable performance of models like the OpenAI o1 can be attributed to their ability to emulate human-like long-time thinking during inference. These models employ extended chain-of-thought (CoT) processes, exploring multiple strategies to enhance problem-solving capabilities. However, a critical question remains: How to intelligently and efficiently scale computational resources during testing. This paper presents the first comprehensive study on the prevalent issue of overthinking in these models, where excessive computational resources are allocated for simple problems with minimal benefit. We introduce novel efficiency metrics from both outcome and process perspectives to evaluate the rational use of computational resources by o1-like models. Using a self-training paradigm, we propose strategies to mitigate overthinking, streamlining reasoning processes without compromising accuracy. Experimental results show that our approach successfully reduces computational overhead while preserving model performance across a range of testsets with varying difficulty levels, such as GSM8K, MATH500, GPQA, and AIME.

## Mulberry: Empowering MLLM with o1-like Reasoning and Reflection via Collective Monte Carlo Tree Search

Github: https://github.com/HJYao00/Mulberry

Paper: https://arxiv.org/abs/2412.18319

Date: 24.12.2024

##### Abstract
In this work, we aim to develop an MLLM that understands and solves questions by learning to create each intermediate step of the reasoning involved till the final answer. To this end, we propose Collective Monte Carlo Tree Search (CoMCTS), a new learning-to-reason method for MLLMs, which introduces the concept of collective learning into ``tree search'' for effective and efficient reasoning-path searching and learning. The core idea of CoMCTS is to leverage collective knowledge from multiple models to collaboratively conjecture, search and identify effective reasoning paths toward correct answers via four iterative operations including Expansion, Simulation and Error Positioning, Backpropagation, and Selection. Using CoMCTS, we construct Mulberry-260k, a multimodal dataset with a tree of rich, explicit and well-defined reasoning nodes for each question. With Mulberry-260k, we perform collective SFT to train our model, Mulberry, a series of MLLMs with o1-like step-by-step Reasoning and Reflection capabilities. Extensive experiments demonstrate the superiority of our proposed methods on various benchmarks. Code will be available at this https URL

## Test-time Computing: from System-1 Thinking to System-2 Thinking

Github: https://github.com/Dereck0602/Awesome_Test_Time_LLMs

Paper: https://arxiv.org/abs/2501.02497

Date: 05.01.2024

##### Abstract
The remarkable performance of the o1 model in complex reasoning demonstrates that test-time computing scaling can further unlock the model's potential, enabling powerful System-2 thinking. However, there is still a lack of comprehensive surveys for test-time computing scaling. We trace the concept of test-time computing back to System-1 models. In System-1 models, test-time computing addresses distribution shifts and improves robustness and generalization through parameter updating, input modification, representation editing, and output calibration. In System-2 models, it enhances the model's reasoning ability to solve complex problems through repeated sampling, self-correction, and tree search. We organize this survey according to the trend of System-1 to System-2 thinking, highlighting the key role of test-time computing in the transition from System-1 models to weak System-2 models, and then to strong System-2 models. We also point out a few possible future directions.
