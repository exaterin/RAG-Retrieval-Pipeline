# Retrieval Evaluation for Code Documentation RAG

*Evaluating Chunking and Retrieval Hyperparameters on the Wikitext Dataset*

## 1. Introduction

This project evaluates a **Retrieval-Augmented Generation (RAG)** ystem by analyzing how various **chunking strategies** and **retrieval depths** influence retrieval quality. The main objective is to ensure that the retriever can return chunks of a corpus that align closely with the golden ground truth excerpts, while balancing **precision (efficiency)** and **recall (coverage)**.


## 2. Dataset and Setup

- **Corpus used**: `wikitexts.md`
- **Queries**: Extracted from `questions_df.csv` with golden references.
- **Embedding model**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Chunking method**: Token-based chunker with overlap using `tiktoken` (FixedTokenChunker).


## 3. Retrieval System Overview

The system consists of the following modules:


| Module               | Class / Function              | Responsibility                                                                 |
|----------------------|-----------------------------|---------------------------------------------------------------------------------|
| `dataset_loader.py`  | DatasetLoader             | Loads the corpus text and questions with golden excerpts            |
| `chunking.py`        | FixedTokenChunker         | Splits text into overlapping token-based chunks using the `tiktoken` tokenizer |
| `embedding.py`       | EmbeddingModel            | Embeds chunks and queries using a pre-trained `SentenceTransformer`                          |
| `retrieval.py`       | RetrievalPipeline         | Retrieves top-k most relevant chunks based on cosine similarity                |
| `evaluation.py`      | RetrievalEvaluator        | Evaluates retrieval quality (precision, recall) using token index overlap |
| `run_pipeline.py`   | `run_pipeline()` + argparse | Orchestrates the full retrieval pipeline: loads data, chunks corpus, generates embeddings, retrieves top-k chunks, evaluates with precision and recall, and saves results to CSV |



## 4. Metric Computation

The evaluation is performed at the **token-level**, based on start-end token ranges.  
We implement:

- **Precision** = correctly retrieved tokens / total retrieved
- **Recall** = correctly retrieved tokens / total golden

All metrics are implemented following the definition in the Chroma benchmark paper.


## 5. Experiment Configuration

We test different values of chunk size and top_k retrieved chunks:

- `chunk_size`: [10, 50, 100, 200, 300, 400]
- `top_k`: [1, 5, 7, 10]



## 7. Visual Analysis

### Precision vs Chunk Size
![Precision vs Chunk Size](results/precision_vs_chunk_size.png)

### Recall vs Chunk Size
![Recall vs Chunk Size](results/recall_vs_chunk_size.png)

---

### Precision Heatmap
![Precision Heatmap](results/precision_heatmap.png)

### Recall Heatmap
![Recall Heatmap](results/recall_heatmap.png)


|   chunk_size |   top_k |   precision |   recall |
|-------------:|--------:|------------:|---------:|
|           10 |       1 |       0.26  |    0.068 |
|           10 |       3 |       0.183 |    0.13  |
|           10 |       5 |       0.142 |    0.167 |
|           10 |       7 |       0.12  |    0.194 |
|           10 |      10 |       0.101 |    0.228 |
|           50 |       1 |       0.237 |    0.272 |
|           50 |       3 |       0.133 |    0.414 |
|           50 |       5 |       0.096 |    0.489 |
|           50 |       7 |       0.075 |    0.519 |
|           50 |      10 |       0.061 |    0.572 |
|          100 |       1 |       0.164 |    0.345 |
|          100 |       3 |       0.083 |    0.499 |
|          100 |       5 |       0.064 |    0.599 |
|          100 |       7 |       0.049 |    0.63  |
|          100 |      10 |       0.037 |    0.679 |
|          200 |       1 |       0.091 |    0.373 |
|          200 |       3 |       0.049 |    0.572 |
|          200 |       5 |       0.034 |    0.645 |
|          200 |       7 |       0.026 |    0.693 |
|          200 |      10 |       0.02  |    0.741 |
|          300 |       1 |       0.061 |    0.357 |
|          300 |       3 |       0.032 |    0.567 |
|          300 |       5 |       0.023 |    0.657 |
|          300 |       7 |       0.018 |    0.694 |
|          300 |      10 |       0.013 |    0.759 |
|          400 |       1 |       0.052 |    0.407 |
|          400 |       3 |       0.026 |    0.587 |
|          400 |       5 |       0.017 |    0.661 |
|          400 |       7 |       0.013 |    0.702 |
|          400 |      10 |       0.01  |    0.765 |