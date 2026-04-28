import argparse
import csv
import os
from typing import Dict, List, Tuple

import pandas as pd
import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

EMBED_PATH = os.path.join(RESULTS_DIR, "review_embeddings.pt")
TRAIN_CSV = os.path.join(DATASET_DIR, "dataset_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "dataset_test.csv")

RETRIEVAL_INDEX_PATH = os.path.join(RESULTS_DIR, "retrieval_index.pt")
RETRIEVAL_EXAMPLES_PATH = os.path.join(RESULTS_DIR, "retrieval_examples.csv")
RETRIEVAL_METRICS_PATH = os.path.join(RESULTS_DIR, "retrieval_metrics.csv")
DECODER_CONTEXTS_PATH = os.path.join(RESULTS_DIR, "decoder_contexts.csv")


def map_sentiment(rating: int) -> int:
    if rating <= 2:
        return 0
    if rating == 3:
        return 1
    return 2


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def parse_k_values(k: int, sweep_k: str) -> List[int]:
    values = {k}
    if sweep_k.strip():
        for item in sweep_k.split(","):
            item = item.strip()
            if item:
                values.add(int(item))
    return sorted(values)


def retrieve_topk(
    query_emb: torch.Tensor,
    train_emb: torch.Tensor,
    k: int,
    query_batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_scores = []
    all_indices = []

    train_emb_t = train_emb.t()
    for start in range(0, query_emb.size(0), query_batch_size):
        end = min(start + query_batch_size, query_emb.size(0))
        sim = torch.matmul(query_emb[start:end], train_emb_t)
        scores, indices = torch.topk(sim, k=k, dim=1)
        all_scores.append(scores.cpu())
        all_indices.append(indices.cpu())

    return torch.cat(all_scores, dim=0), torch.cat(all_indices, dim=0)


def sentiment_agreement_at_k(
    query_labels: torch.Tensor,
    retrieved_indices: torch.Tensor,
    train_labels: torch.Tensor,
) -> float:
    retrieved_labels = train_labels[retrieved_indices]
    agreement = (retrieved_labels == query_labels.unsqueeze(1)).float().mean().item()
    return agreement


def save_metrics(rows: List[Dict[str, float]]) -> None:
    with open(RETRIEVAL_METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["k", "sentiment_agreement_at_k", "mean_top1_cosine"])
        writer.writeheader()
        writer.writerows(rows)


def save_examples(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    num_examples: int,
) -> None:
    num_examples = min(num_examples, len(test_df))
    rows = []
    for q_idx in range(num_examples):
        query_text = str(test_df.iloc[q_idx]["reviewText"])
        query_rating = int(test_df.iloc[q_idx]["overall"])
        for rank in range(k):
            t_idx = int(indices[q_idx, rank].item())
            rows.append(
                {
                    "query_index": q_idx,
                    "query_rating": query_rating,
                    "query_text": query_text,
                    "rank": rank + 1,
                    "retrieved_train_index": t_idx,
                    "retrieved_similarity": float(scores[q_idx, rank].item()),
                    "retrieved_rating": int(train_df.iloc[t_idx]["overall"]),
                    "retrieved_text": str(train_df.iloc[t_idx]["reviewText"]),
                }
            )
    pd.DataFrame(rows).to_csv(RETRIEVAL_EXAMPLES_PATH, index=False)


def build_decoder_context(train_df: pd.DataFrame, retrieved_indices: torch.Tensor, query_idx: int) -> str:
    passages = []
    for idx in retrieved_indices[query_idx].tolist():
        passages.append(str(train_df.iloc[int(idx)]["reviewText"]))
    return " <CTX_SEP> ".join(passages)


def save_decoder_contexts(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    retrieved_indices: torch.Tensor,
) -> None:
    rows = []
    for q_idx in range(len(test_df)):
        rows.append(
            {
                "query_index": q_idx,
                "query_rating": int(test_df.iloc[q_idx]["overall"]),
                "query_text": str(test_df.iloc[q_idx]["reviewText"]),
                "decoder_context": build_decoder_context(train_df, retrieved_indices, q_idx),
            }
        )
    pd.DataFrame(rows).to_csv(DECODER_CONTEXTS_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part B retrieval module")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieval size")
    parser.add_argument("--sweep_k", type=str, default="3,5,10", help="Comma-separated k values for analysis")
    parser.add_argument("--query_batch_size", type=int, default=256)
    parser.add_argument("--num_examples", type=int, default=10, help="How many qualitative examples to export")
    parser.add_argument("--max_queries", type=int, default=0, help="Use >0 for quick smoke tests")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(f"Missing embeddings at {EMBED_PATH}. Run Part_A.py first.")
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        raise FileNotFoundError("Missing Dataset CSVs. Run Dataset_prepare.py first.")

    emb = torch.load(EMBED_PATH, map_location="cpu")
    train_emb = normalize_rows(emb["train"].float())
    test_emb = normalize_rows(emb["test"].float())

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if args.max_queries > 0:
        limit = min(args.max_queries, test_emb.size(0))
        test_emb = test_emb[:limit]
        test_df = test_df.iloc[:limit].reset_index(drop=True)

    k_values = parse_k_values(args.k, args.sweep_k)
    k_max = max(k_values)
    scores_max, indices_max = retrieve_topk(test_emb, train_emb, k_max, args.query_batch_size)

    train_sent = torch.tensor(train_df["overall"].apply(map_sentiment).tolist(), dtype=torch.long)
    test_sent = torch.tensor(test_df["overall"].apply(map_sentiment).tolist(), dtype=torch.long)

    metric_rows = []
    for k in k_values:
        scores_k = scores_max[:, :k]
        indices_k = indices_max[:, :k]
        agreement = sentiment_agreement_at_k(test_sent, indices_k, train_sent)
        metric_rows.append(
            {
                "k": k,
                "sentiment_agreement_at_k": agreement,
                "mean_top1_cosine": float(scores_k[:, 0].mean().item()),
            }
        )

    save_metrics(metric_rows)

    final_scores = scores_max[:, : args.k]
    final_indices = indices_max[:, : args.k]

    torch.save(
        {
            "k": args.k,
            "scores": final_scores,
            "indices": final_indices,
            "query_count": final_scores.size(0),
        },
        RETRIEVAL_INDEX_PATH,
    )

    save_examples(test_df, train_df, final_scores, final_indices, args.k, args.num_examples)
    save_decoder_contexts(test_df, train_df, final_indices)

    sample_context = build_decoder_context(train_df, final_indices, query_idx=0)
    print(f"Stored retrieval index at: {RETRIEVAL_INDEX_PATH}")
    print(f"Stored retrieval metrics at: {RETRIEVAL_METRICS_PATH}")
    print(f"Stored retrieval examples at: {RETRIEVAL_EXAMPLES_PATH}")
    print(f"Stored decoder contexts at: {DECODER_CONTEXTS_PATH}")
    print(f"Sample decoder context length: {len(sample_context)}")


if __name__ == "__main__":
    main()
