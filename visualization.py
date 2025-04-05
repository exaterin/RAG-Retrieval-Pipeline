import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from run_pipeline import run_pipeline

# Configurable ranges
CHUNK_SIZES = [10, 50, 100, 200, 300, 400]
TOP_K_VALUES = [1, 3, 5, 7, 10]
CORPUS_ID = "wikitexts"
MODEL_NAME = "all-MiniLM-L6-v2"
OVERLAP_RATIO = 0.2  # 20% overlap

def run_experiment(chunk_size, top_k):
    print(f"Running experiment with chunk_size={chunk_size}, top_k={top_k}")
    
    # Run the pipeline and get a DataFrame with precision and recall
    result_df = run_pipeline(
        corpus_id=CORPUS_ID,
        chunk_size=chunk_size,
        overlap=OVERLAP_RATIO,
        top_k=top_k,
        model_name=MODEL_NAME,
        return_df=True
    )
    
    # Annotate with chunk_size and top_k for grouping
    result_df["chunk_size"] = chunk_size
    result_df["top_k"] = top_k
    return result_df


def main():
    all_results = []

    os.makedirs('results', exist_ok=True)

    for chunk_size in CHUNK_SIZES:
        for top_k in TOP_K_VALUES:
            result_df = run_experiment(chunk_size, top_k)
            all_results.append(result_df)

    # Combine and save
    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv("results/evaluation_results.csv", index=False)

    summary = full_df.groupby(["chunk_size", "top_k"]).agg({
        "precision": "mean",
        "recall": "mean"
    }).reset_index()
    summary.to_csv("results/evaluation_summary.csv", index=False)

    # Precision vs. Chunk Size
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x="chunk_size", y="precision", hue="top_k", marker="o")
    plt.title("Precision vs. Chunk Size (Grouped by top_k)")
    plt.xlabel("Chunk Size")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig("results/precision_vs_chunk_size.png")
    plt.clf()

    # Recall vs. Chunk Size
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=summary, x="chunk_size", y="recall", hue="top_k", marker="o")
    plt.title("Recall vs. Chunk Size (Grouped by top_k)")
    plt.xlabel("Chunk Size")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.savefig("results/recall_vs_chunk_size.png")
    plt.clf()

    # Heatmaps
    plt.figure(figsize=(8, 6))
    sns.heatmap(summary.pivot("chunk_size", "top_k", "precision"), annot=True, cmap="Blues")
    plt.title("Precision Heatmap")
    plt.savefig("results/precision_heatmap.png")
    plt.clf()

    plt.figure(figsize=(8, 6))
    sns.heatmap(summary.pivot("chunk_size", "top_k", "recall"), annot=True, cmap="Greens")
    plt.title("Recall Heatmap")
    plt.savefig("results/recall_heatmap.png")
    plt.clf()

    print("All experiments complete.")


if __name__ == "__main__":
    main()
