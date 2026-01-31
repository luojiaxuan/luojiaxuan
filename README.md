<div align="center">

# Hi there, I'm Jiaxuan (Jaxan) Luo 👋

### Streaming Speech Foundation Models | RL Alignment (GRPO) | System-Aware Inference

[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=bdstD5kAAAAJ&hl=en)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jiaxuanluo/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:luojiaxuan1215@gmail.com)

</div>

---

### 🚀 About Me

I am a **Research Engineer** specializing in **Reinforcement Learning (RL)**, **Multimodal Systems**, and **Post-training Optimization**.

I bridge the gap between **SOTA Algorithms** and **High-Performance Systems**. My work focuses on aligning Audio/Speech Foundation Models using RL (GRPO) and optimizing inference throughput using kernel-level techniques (FlashInfer, vLLM).

- 🎓 **Education:** M.S. in CS from **Johns Hopkins University**; Currently Research Assistant at **CMU LTI**.
- 🔭 **Currently working on:** Streaming Audio-Text RAG & Policy Optimization for Latency-Constrained Generation.

---

### 📝 Selected Research & Projects

> **Note:** Code for papers under review will be released upon acceptance.

#### 🎙️ [RASST: Cross-modal Retrieval-Augmented Speech Translation]
**First Author** | *Submitted to ACL 2026*
- Proposed a streaming Cross-modal RAG framework achieving **89.3% terminology accuracy** (+16.3% vs baselines).
- Engineered a **policy-aware decoding strategy** to handle asynchronous retrieval, maintaining **1.7s StreamLAAL**.
- **Stack:** PyTorch, FAISS, Cross-Modal Contrastive Learning.

#### 🧠 [Hierarchical Policy Optimization (GRPO) for Speech]
**Co-Developer** | *Submitted to ACL 2026*
- Co-developed a streaming alignment framework based on **Group Relative Policy Optimization (GRPO)**.
- Eliminated the Value Network to reduce memory usage by **40%** while stabilizing reward convergence.
- **Stack:** RLHF, PPO/GRPO, DeepSpeed.

#### ⚡ [High-Throughput Streaming Inference Engine]
**Core Contributor** | *Built on InfiniSST (ACL 2025)*
- Architected a custom inference engine on **Ray** with **Paged Attention via FlashInfer kernels**.
- Scaled the prototype to **32 concurrent sessions per GPU** with sub-200ms serving overhead.
- **Stack:** Ray, FlashInfer, CUDA, Python.

#### 🛡️ [Is Vibe Coding Safe? Agent Vulnerability Benchmark]
**Co-Author** | *Submitted to ICML 2026* | [arXiv:2512.03262](https://arxiv.org/abs/2512.03262)
- Benchmarked vulnerability scenarios for Agentic Code Generation.
- Deployed **Kimi K2 (MoE)** using **vLLM with Tensor Parallelism**.

---

### 🛠️ Technical Arsenal

<div align="left">

| Domain | Stack |
| :--- | :--- |
| **Model Training** | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![DeepSpeed](https://img.shields.io/badge/DeepSpeed-000000?style=flat-square&logo=deepspeed&logoColor=white) ![Megatron-LM](https://img.shields.io/badge/Megatron--LM-76B900?style=flat-square) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black) |
| **Inference & Sys** | ![vLLM](https://img.shields.io/badge/vLLM-000000?style=flat-square) ![Ray](https://img.shields.io/badge/Ray-028CF0?style=flat-square&logo=ray&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white) ![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white) |
| **Algorithms** | ![RLHF](https://img.shields.io/badge/RLHF-FF6F00?style=flat-square) ![GRPO](https://img.shields.io/badge/GRPO-4285F4?style=flat-square) ![RAG](https://img.shields.io/badge/Multimodal%20RAG-0F9D58?style=flat-square) |
| **Languages** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=c%2B%2B&logoColor=white) ![Shell](https://img.shields.io/badge/Shell-4EAA25?style=flat-square&logo=gnu-bash&logoColor=white) |

</div>

---

### 💼 Experience

- **TikTok (San Jose)** | *Machine Learning Engineer (GenAI & Post-Training)*
    - Engineered analytics pipeline using **DeepSeek R1 (70B)** with schema-aware CoT.
    - Optimized Multi-LoRA serving via Ray + vLLM.

- **Alibaba Group (Hangzhou)** | *Machine Learning Engineer II*
    - Built distributed serving systems (Java/C++) handling **80,000 QPS**.
    - Developed RL policies for marketplace ROI optimization.

---
