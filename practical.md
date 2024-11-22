1. OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs
2. SambaMixer: State of Health Prediction of Li-ion Batteries using Mamba State Space Models


## OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs

Github: https://github.com/AkariAsai/OpenScholar

Paper: https://arxiv.org/pdf/2411.14199

Date: 21.11.2024

##### Abstract
Scientific progress depends on researchers' ability to synthesize the growing body of literature. Can large language models (LMs) assist scientists in this task? We introduce OpenScholar, a specialized retrieval-augmented LM that answers scientific queries by identifying relevant passages from 45 million open-access papers and synthesizing citation-backed responses. To evaluate OpenScholar, we develop ScholarQABench, the first large-scale multi-domain benchmark for literature search, comprising 2,967 expert-written queries and 208 long-form answers across computer science, physics, neuroscience, and biomedicine. On ScholarQABench, OpenScholar-8B outperforms GPT-4o by 5% and PaperQA2 by 7% in correctness, despite being a smaller, open model. While GPT4o hallucinates citations 78 to 90% of the time, OpenScholar achieves citation accuracy on par with human experts. OpenScholar's datastore, retriever, and self-feedback inference loop also improves off-the-shelf LMs: for instance, OpenScholar-GPT4o improves GPT-4o's correctness by 12%. In human evaluations, experts preferred OpenScholar-8B and OpenScholar-GPT4o responses over expert-written ones 51% and 70% of the time, respectively, compared to GPT4o's 32%. We open-source all of our code, models, datastore, data and a public demo.

## SambaMixer: State of Health Prediction of Li-ion Batteries using Mamba State Space Models

Github: https://github.com/sascha-kirch/samba-mixer

Paper: https://arxiv.org/pdf/2411.00233

Date: 31.10.2024

##### Abstract
The state of health (SOH) of a Li-ion battery is a critical parameter that determines the remaining capacity and the remaining lifetime of the battery. In this paper, we propose SambaMixer a novel structured state space model (SSM) for predicting the state of health of Li-ion batteries. The proposed SSM is based on the MambaMixer architecture, which is designed to handle multi-variate time signals. We evaluate our model on the NASA battery discharge dataset and show that our model outperforms the state-of-the-art on this dataset. We further introduce a novel anchor-based resampling method which ensures time signals are of the expected length while also serving as augmentation technique. Finally, we condition prediction on the sample time and the cycle time difference using positional encodings to improve the performance of our model and to learn recuperation effects. Our results proof that our model is able to predict the SOH of Li-ion batteries with high accuracy and robustness.
