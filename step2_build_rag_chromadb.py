import time
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. Sample Technical Document Data
SAMPLE_TECH_DOC = """
The Quantum-X System is an advanced AI compute engine developed in 2025.
It operates on a custom 4nm architecture designed specifically for high-throughput tensor operations.
The system features a peak memory bandwidth of 3.2 TB/s and consumes 450 Watts under max load.
For safety and operational constraints, the maximum operating temperature is 85 degrees Celsius.
If temperature exceeds 85°C, the thermal throttling mechanism automatically reduces clock speed by 40%.
The primary software stack used to interface with Quantum-X is the Q-Framework v2.4, which supports native PyTorch and ONNX execution.
"""

def setup_vectorstore():
    print("\nProcessing technical documentation and building vector store...")
    docs = [Document(page_content=SAMPLE_TECH_DOC, metadata={"source": "quantum_x_spec.txt"})]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    split_docs = text_splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embedding_model)
    print(f"Success! Chunked document into {len(split_docs)} pieces and stored in ChromaDB.")
    return vectorstore

def run_rag_inference(vectorstore, query="What happens if Quantum-X temperature goes above 85 degrees Celsius?"):
    print("\nLoading Qwen2.5-0.5B-Instruct lightweight LLM...")
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=pipe)

    print(f"Executing Query: '{query}'")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    rag_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    start_time = time.time()
    response = llm.invoke(rag_prompt)
    latency = time.time() - start_time

    print("\n" + "="*50)
    print("RAG INFERENCE RESPONSE:")
    print("="*50)
    print(response.strip())
    print("="*50)
    print(f"RAG Inference Latency: {latency:.2f} seconds")
    return latency

if __name__ == "__main__":
    vstore = setup_vectorstore()
    run_rag_inference(vstore)
