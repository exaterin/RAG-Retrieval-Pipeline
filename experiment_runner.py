import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from chunking import FixedTokenChunker
from embedding import EmbeddingModel
from retrieval import RetrievalPipeline
from evaluation import RetrievalEvaluator

# Define your experiment parameters
CHUNK_SIZES = [10, 50, 200, 300, 400]
TOP_K_VALUES = [1, 5, 10]
CORPUS_ID = "wikitexts"
TEXT_FILE = f"corpora/{CORPUS_ID}.md"
QUESTIONS_FILE = "questions_df.csv"

def run_experiment(chunk_size, top_k):
    with open(TEXT_FILE, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    questions_df = pd.read_csv(QUESTIONS_FILE)
    filtered_questions_df = questions_df[questions_df['corpus_id'] == CORPUS_ID]

    chunker = FixedTokenChunker(encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=int(chunk_size // 5))
    embedder = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    retrieval_pipeline = RetrievalPipeline(embedder)
    evaluator = RetrievalEvaluator(filtered_questions_df, document=corpus)

    # Chunk the corpus
    corpus_chunks = chunker.split_text(corpus)
    corpus_embeddings = embedder.embed(corpus_chunks)

    results = []

    # Iterate over each question
    for index, row in filtered_questions_df.iterrows():
        query = row['question']

        retrieved_data = retrieval_pipeline.retrieve(query, corpus_chunks, corpus_embeddings, top_k=top_k)
        retrieved_chunks = [chunk if isinstance(chunk, str) else chunk[0] for chunk in retrieved_data]

        evaluation = evaluator.evaluate(query, retrieved_chunks, CORPUS_ID)
        precision, recall = evaluation["precision"], evaluation["recall"]

        results.append({
            'query': query,
            'chunk_size': chunk_size,
            'top_k': top_k,
            'precision': precision,
            'recall': recall
        })

    return pd.DataFrame(results)


def main():
    all_results = []

    os.makedirs('results', exist_ok=True)

    # Run experiments for each chunk_size and top_k combination
    for chunk_size in CHUNK_SIZES:
        for top_k in TOP_K_VALUES:
            print(f"Running experiment with chunk_size={chunk_size}, top_k={top_k}")
            result_df = run_experiment(chunk_size, top_k)
            all_results.append(result_df)

    # Combine all results
    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv("results/evaluation_results.csv", index=False)

    # Aggregate
    summary = full_df.groupby(["chunk_size", "top_k"]).agg({
        "precision": "mean",
        "recall": "mean"
    }).reset_index()
    summary.to_csv("results/evaluation_summary.csv", index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x="chunk_size", y="precision", hue="top_k", marker="o")
    plt.title("Precision vs. Chunk Size (Grouped by top_k)")
    plt.xlabel("Chunk Size")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig("results/precision_vs_chunk_size.png")
    plt.clf()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x="chunk_size", y="recall", hue="top_k", marker="o")
    plt.title("Recall vs. Chunk Size (Grouped by top_k)")
    plt.xlabel("Chunk Size")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.savefig("results/recall_vs_chunk_size.png")
    plt.clf()


    # Heatmap Plot
    pivot_table = summary.pivot(index="chunk_size", columns="top_k", values="precision")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="Blues")
    plt.title("Precision Heatmap (Chunk Size vs. top_k)")
    plt.savefig("results/precision_heatmap.png")
    plt.clf()

    pivot_table = summary.pivot(index="chunk_size", columns="top_k", values="recall")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, cmap="Greens")
    plt.title("Recall Heatmap (Chunk Size vs. top_k)")
    plt.savefig("results/recall_heatmap.png")
    plt.clf()

    print("Done.")

if __name__ == "__main__":
    main()
