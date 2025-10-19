# Large Language Model Assignments

This repository contains computer assignments from University of Tehran's graduate course on Large Language Models (LLMs), demonstrating a wide range of skills in LLM engineering, fine-tuning, alignment, and application development. The projects cover the entire lifecycle of building with modern LLMs, from foundational techniques to complex, stateful agentic systems.

## Key Topics & Features

-   **Efficient Fine-Tuning (PEFT):** Implemented and evaluated Parameter-Efficient Fine-Tuning techniques, including **LoRa** for text classification and **QLoRA** for fine-tuning Gemma models on Text-to-SQL tasks.
-   **Human Preference Alignment:** Developed pipelines for aligning LLMs using state-of-the-art, RL-free methods like **Direct Preference Optimization (DPO)** and **Odds Ratio Preference Optimization (ORPO)**, leveraging the `unsloth` library for high-performance training.
-   **Retrieval-Augmented Generation (RAG):** Built and evaluated end-to-end RAG systems using `LangChain`. Explored both sparse (TF-IDF) and dense (FAISS with `bge-small-en`) retrievers, with automated evaluation performed using the `Ragas` framework.
-   **LLM-Powered Agents & Graphs:** Designed and implemented multi-step Text-to-SQL pipelines using `LangGraph`, including a difficulty-based router and a ReAct agent that uses tools to interact with database schemas.
-   **Model Optimization:** Applied 4-bit quantization using `bitsandbytes` to reduce model memory footprint by over 75%, enabling the fine-tuning of larger models on consumer-grade hardware.
-   **LLM Evaluation & Explainability:** Used LLMs as judges to generate preference data and explored self-explanation techniques (Explain-then-Predict vs. Predict-and-Explain) to analyze model reasoning.

## Project Structure

The repository is organized into four main assignments, each focusing on a distinct area of LLM development:

-   `CA1/`: **Foundations & LoRa Fine-Tuning** - Covers basic LLM interaction, tokenization, and fine-tuning a Llama-3 model for emotion classification.
-   `CA2/`: **Preference Alignment with DPO & ORPO** - Explores in-context learning, reward modeling, and aligns a Llama-3 model using DPO and ORPO.
-   `CA3/`: **RAG & LLM-as-a-Judge** - Focuses on building RAG pipelines for recipe generation and using LLMs for automated evaluation and preference data creation.
-   `CA4/`: **Quantization & Advanced Text-to-SQL Agents** - Details QLoRA fine-tuning for a Text-to-SQL task and constructs advanced, multi-step agentic systems with `LangGraph`.

## Technologies & Frameworks

-   **Core Libraries:** PyTorch, Transformers, Datasets, PEFT, TRL
-   **LLM Frameworks:** LangChain, LangGraph
-   **Optimization:** bitsandbytes, unsloth
-   **Retrieval:** FAISS, Sentence Transformers
-   **Evaluation:** Ragas, Scikit-learn
-   **Models:** Llama-3.2, Phi-3, Gemma, Qwen, StableLM, and others.
