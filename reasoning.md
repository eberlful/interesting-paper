1. Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions
2. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models


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
