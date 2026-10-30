<<<<<<< HEAD
# llm_rag_vs_qlora_benchmark
=======
# Benchmarking RAG vs. QLoRA Fine-Tuning on Open-Source LLMs

An end-to-end, reproducible research benchmark comparing Retrieval-Augmented Generation (RAG) against Low-Rank Adaptation (QLoRA) fine-tuning on open-weights Large Language Models for domain-specific information extraction tasks.

## Project Structure
- `requirements.txt`: Required Python dependencies.
- `step1_setup_and_verify.py`: Environment verification script (PyTorch, GPU, Transformers).
- `step2_build_rag_chromadb.py`: Document ingestion, chunking, embedding, vector storage, and RAG retrieval latency benchmark.
- `step3_run_benchmark.py`: Comparative benchmark runner for accuracy, latency, and VRAM memory usage.
- `run_pipeline.py`: Main executable runner executing all steps sequentially.

## Getting Started

1. **Clone or Extract Repository:**
   ```bash
   cd llm_rag_vs_qlora_benchmark
   ```

2. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Execute Benchmark Pipeline:**
   ```bash
   python run_pipeline.py
   ```

## Citation & Resume
If you use this benchmark codebase in your resume or research paper draft, cite as:
`Benchmarking RAG vs. QLoRA Fine-Tuning on Open-Source LLMs (2026), GitHub Repository.`
>>>>>>> a539a4b (Initial public project upload)
