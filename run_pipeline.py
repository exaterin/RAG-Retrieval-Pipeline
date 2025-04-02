import os
import pandas as pd
from chunking import FixedTokenChunker
from embedding import EmbeddingModel
from retrieval import RetrievalPipeline
from evaluation import RetrievalEvaluator


def main():
    corpus_id = "wikitexts"
    text_file = f"corpora/{corpus_id}.md"
    questions_file = "questions_df.csv"

    with open(text_file, 'r', encoding='utf-8') as f:
        corpus = f.read()

    questions_df = pd.read_csv(questions_file)
    filtered_questions_df = questions_df[questions_df['corpus_id'] == corpus_id]

    chunker = FixedTokenChunker(encoding_name="cl100k_base", chunk_size=400, chunk_overlap=100)
    embedder = EmbeddingModel(model_name="multi-qa-mpnet-base-dot-v1")
    retrieval_pipeline = RetrievalPipeline(embedder)
    evaluator = RetrievalEvaluator(filtered_questions_df, document=corpus)

    # Chunk the corpus
    corpus_chunks = chunker.split_text(corpus)
    corpus_embeddings = embedder.embed(corpus_chunks)

    results = []


    # Iterate over each question
    for index, row in filtered_questions_df.iterrows():
        query = row['question']

        retrieved_data = retrieval_pipeline.retrieve(query, corpus_chunks, corpus_embeddings, top_k=5)
        retrieved_chunks = [chunk if isinstance(chunk, str) else chunk[0] for chunk in retrieved_data]

        evaluation = evaluator.evaluate(query, retrieved_chunks, corpus_id)
        precision, recall = evaluation["precision"], evaluation["recall"]


        results.append({
            'query': query,
            'precision': precision,
            'recall': recall
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv('results/all_results.csv', index=False)


if __name__ == "__main__":
    main()
