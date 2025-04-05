import os
import argparse
import pandas as pd

from chunking import FixedTokenChunker
from embedding import EmbeddingModel
from retrieval import RetrievalPipeline
from evaluation import RetrievalEvaluator


def run_pipeline(corpus_id: str, chunk_size: int, overlap: float, top_k: int, model_name: str, return_df=False):
    text_file = f"data/{corpus_id}.md"
    questions_file = "data/questions_df.csv"

    with open(text_file, 'r', encoding='utf-8') as f:
        corpus = f.read()

    questions_df = pd.read_csv(questions_file)
    filtered_questions_df = questions_df[questions_df['corpus_id'] == corpus_id]

    chunk_overlap = int(chunk_size * overlap)
    chunker = FixedTokenChunker(encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = EmbeddingModel(model_name=model_name)
    retrieval_pipeline = RetrievalPipeline(embedder)
    evaluator = RetrievalEvaluator(filtered_questions_df, document=corpus)

    # Chunk the corpus
    corpus_chunks = chunker.split_text(corpus)
    corpus_embeddings = embedder.embed(corpus_chunks)

    results = []

    for index, row in filtered_questions_df.iterrows():
        query = row['question']
        retrieved_data = retrieval_pipeline.retrieve(query, corpus_chunks, corpus_embeddings, top_k=top_k)
        retrieved_chunks = [chunk if isinstance(chunk, str) else chunk[0] for chunk in retrieved_data]

        evaluation = evaluator.evaluate(query, retrieved_chunks, corpus_id)
        precision, recall = evaluation["precision"], evaluation["recall"]

        results.append({
            'query': query,
            'precision': precision,
            'recall': recall
        })

    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = f"results/results_{corpus_id}_cs{chunk_size}_ol{chunk_overlap}_top{top_k}.csv"
    
    results_df = pd.DataFrame(results)
    if return_df:
        return results_df
    else:
        output_path = f"results/results_{corpus_id}_cs{chunk_size}_ol{chunk_overlap}_top{top_k}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation pipeline.")

    parser.add_argument("--corpus_id", type=str, default="wikitexts", help="Name of the corpus")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chunk size in tokens")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio (0.0 to 1.0)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top retrieved chunks")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Hugging Face model name for embeddings")

    args = parser.parse_args()

    run_pipeline(
        corpus_id=args.corpus_id,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        top_k=args.top_k,
        model_name=args.model_name
    )

