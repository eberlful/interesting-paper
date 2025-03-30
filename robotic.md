1. One to rule them all: natural language to bind communication, perception and action
2. Spatially Visual Perception for End-to-End Robotic Learning
3. 3D-VLA: A 3D Vision-Language-Action Generative World Model
4. Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids
5. Gemini Robotics: Bringing AI into the Physical World


## One to rule them all: natural language to bind communication, perception and action

Paper: https://arxiv.org/pdf/2411.15033

Date: 22.11.2024

##### Abstract
In recent years, research in the area of human-robot interaction has focused on developing robots capable of understanding complex human instructions and performing tasks in dynamic and diverse environments. These systems have a wide range of applications, from personal assistance to industrial robotics, emphasizing the importance of robots interacting flexibly, naturally and safely with humans. This paper presents an advanced architecture for robotic action planning that integrates communication, perception, and planning with Large Language Models (LLMs). Our system is designed to translate commands expressed in natural language into executable robot actions, incorporating environmental information and dynamically updating plans based on real-time feedback. The Planner Module is the core of the system where LLMs embedded in a modified ReAct framework are employed to interpret and carry out user commands. By leveraging their extensive pre-trained knowledge, LLMs can effectively process user requests without the need to introduce new knowledge on the changing environment. The modified ReAct framework further enhances the execution space by providing real-time environmental perception and the outcomes of physical actions. By combining robust and dynamic semantic map representations as graphs with control components and failure explanations, this architecture enhances a robot adaptability, task execution, and seamless collaboration with human users in shared and dynamic environments. Through the integration of continuous feedback loops with the environment the system can dynamically adjusts the plan to accommodate unexpected changes, optimizing the robot ability to perform tasks. Using a dataset of previous experience is possible to provide detailed feedback about the failure. Updating the LLMs context of the next iteration with suggestion on how to overcame the issue.

## Spatially Visual Perception for End-to-End Robotic Learning

Paper: https://arxiv.org/pdf/2411.17458

Date: 26.11.2024

##### Abstract
Recent advances in imitation learning have shown significant promise for robotic control and embodied intelligence. However, achieving robust generalization across diverse mounted camera observations remains a critical challenge. In this paper, we introduce a video-based spatial perception framework that leverages 3D spatial representations to address environmental variability, with a focus on handling lighting changes. Our approach integrates a novel image augmentation technique, AugBlender, with a state-of-the-art monocular depth estimation model trained on internet-scale data. Together, these components form a cohesive system designed to enhance robustness and adaptability in dynamic scenarios. Our results demonstrate that our approach significantly boosts the success rate across diverse camera exposures, where previous models experience performance collapse. Our findings highlight the potential of video-based spatial perception models in advancing robustness for end-to-end robotic learning, paving the way for scalable, low-cost solutions in embodied intelligence.

## 3D-VLA: A 3D Vision-Language-Action Generative World Model

Github: https://vis-www.cs.umass.edu/3dvla/

Paper: https://arxiv.org/abs/2403.09631

Date: 14.03.2024

##### Abstract
Recent vision-language-action (VLA) models rely on 2D inputs, lacking integration with the broader realm of the 3D physical world. Furthermore, they perform action prediction by learning a direct mapping from perception to action, neglecting the vast dynamics of the world and the relations between actions and dynamics. In contrast, human beings are endowed with world models that depict imagination about future scenarios to plan actions accordingly. To this end, we propose 3D-VLA by introducing a new family of embodied foundation models that seamlessly link 3D perception, reasoning, and action through a generative world model. Specifically, 3D-VLA is built on top of a 3D-based large language model (LLM), and a set of interaction tokens is introduced to engage with the embodied environment. Furthermore, to inject generation abilities into the model, we train a series of embodied diffusion models and align them into the LLM for predicting the goal images and point clouds. To train our 3D-VLA, we curate a large-scale 3D embodied instruction dataset by extracting vast 3D-related information from existing robotics datasets. Our experiments on held-in datasets demonstrate that 3D-VLA significantly improves the reasoning, multimodal generation, and planning capabilities in embodied environments, showcasing its potential in real-world applications.

## Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids

Github: https://toruowo.github.io/recipe/

Paper: https://arxiv.org/abs/2502.20396

Date: 27.02.2025

##### Abstract
Reinforcement learning has delivered promising results in achieving human- or even superhuman-level capabilities across diverse problem domains, but success in dexterous robot manipulation remains limited. This work investigates the key challenges in applying reinforcement learning to solve a collection of contact-rich manipulation tasks on a humanoid embodiment. We introduce novel techniques to overcome the identified challenges with empirical validation. Our main contributions include an automated real-to-sim tuning module that brings the simulated environment closer to the real world, a generalized reward design scheme that simplifies reward engineering for long-horizon contact-rich manipulation tasks, a divide-and-conquer distillation process that improves the sample efficiency of hard-exploration problems while maintaining sim-to-real performance, and a mixture of sparse and dense object representations to bridge the sim-to-real perception gap. We show promising results on three humanoid dexterous manipulation tasks, with ablation studies on each technique. Our work presents a successful approach to learning humanoid dexterous manipulation using sim-to-real reinforcement learning, achieving robust generalization and high performance without the need for human demonstration.

## Gemini Robotics: Bringing AI into the Physical World

Paper: https://arxiv.org/abs/2503.20020

Date: 25.03.2025

##### Abstract
Recent advancements in large multimodal models have led to the emergence of remarkable generalist capabilities in digital domains, yet their translation to physical agents such as robots remains a significant challenge. This report introduces a new family of AI models purposefully designed for robotics and built upon the foundation of Gemini 2.0. We present Gemini Robotics, an advanced Vision-Language-Action (VLA) generalist model capable of directly controlling robots. Gemini Robotics executes smooth and reactive movements to tackle a wide range of complex manipulation tasks while also being robust to variations in object types and positions, handling unseen environments as well as following diverse, open vocabulary instructions. We show that with additional fine-tuning, Gemini Robotics can be specialized to new capabilities including solving long-horizon, highly dexterous tasks, learning new short-horizon tasks from as few as 100 demonstrations and adapting to completely novel robot embodiments. This is made possible because Gemini Robotics builds on top of the Gemini Robotics-ER model, the second model we introduce in this work. Gemini Robotics-ER (Embodied Reasoning) extends Gemini's multimodal reasoning capabilities into the physical world, with enhanced spatial and temporal understanding. This enables capabilities relevant to robotics including object detection, pointing, trajectory and grasp prediction, as well as multi-view correspondence and 3D bounding box predictions. We show how this novel combination can support a variety of robotics applications. We also discuss and address important safety considerations related to this new class of robotics foundation models. The Gemini Robotics family marks a substantial step towards developing general-purpose robots that realizes AI's potential in the physical world.
