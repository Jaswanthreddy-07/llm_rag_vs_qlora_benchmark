# Benchmarking RAG vs. QLoRA Fine-Tuning on Open-Source LLMs for Automated Information Extraction

An end-to-end, reproducible AI research benchmark evaluating the trade-offs between **Retrieval-Augmented Generation (RAG)** and **Quantized Low-Rank Adaptation (QLoRA)** fine-tuning on open-weights Large Language Models for domain-specific information extraction tasks.

---

## 📌 Project Overview
When adapting LLMs for enterprise information extraction from unstructured technical documentation, system designers face a key architectural decision: dynamic context injection via RAG or model parameter adaptation via fine-tuning. 

This repository provides a complete evaluation pipeline measuring:
- **Inference Latency** (CPU vs. GPU execution)
- **Factual Extraction Accuracy** (Zero-shot performance)
- **Memory Overhead & Resource Footprint**

---

## 🛠️ Tech Stack & Frameworks
- **Language:** Python 3.10+
- **LLM Engine:** Hugging Face `transformers`, `accelerate`, `bitsandbytes`
- **Models:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Vector Database:** `ChromaDB`
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Orchestration:** `LangChain` / `langchain-chroma`
- **Publication Format:** IEEE Double-Column LaTeX

---

## 📂 Repository Structure

```text
llm-rag-vs-qlora-benchmark/
├── paper/
│   └── paper.pdf                # Compiled IEEE research paper manuscript
├── main.py                      # Standalone combined Python pipeline script
├── step1_setup_and_verify.py    # Hardware and library initialization script
├── step2_build_rag_chromadb.py  # Vector store creation & RAG execution script
├── step3_run_benchmark.py       # Metrics logger & baseline analysis
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
