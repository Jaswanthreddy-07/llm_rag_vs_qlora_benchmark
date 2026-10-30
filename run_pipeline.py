from step1_setup_and_verify import verify_environment
from step2_build_rag_chromadb import setup_vectorstore, run_rag_inference
from step3_run_benchmark import benchmark_summary

def main():
    print("\n>>> STARTING RAG VS QLORA BENCHMARK PIPELINE <<<\n")
    verify_environment()
    vstore = setup_vectorstore()
    run_rag_inference(vstore)
    benchmark_summary()
    print("\n>>> PIPELINE COMPLETE <<<\n")

if __name__ == "__main__":
    main()
