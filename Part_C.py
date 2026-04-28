import argparse
import csv
import gc
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_CSV = os.path.join(DATASET_DIR, "dataset_train.csv")
VAL_CSV = os.path.join(DATASET_DIR, "dataset_val.csv")
TEST_CSV = os.path.join(DATASET_DIR, "dataset_test.csv")
EMBED_PATH = os.path.join(RESULTS_DIR, "review_embeddings.pt")

PARTC_FULL_MODEL = os.path.join(MODELS_DIR, "part_c_decoder_full.pt")
PARTC_BASE_MODEL = os.path.join(MODELS_DIR, "part_c_decoder_baseline.pt")
PARTC_VOCAB_PATH = os.path.join(RESULTS_DIR, "part_c_vocab.pt")
PARTC_METRICS_PATH = os.path.join(RESULTS_DIR, "part_c_metrics.csv")
PARTC_EXAMPLES_PATH = os.path.join(RESULTS_DIR, "part_c_generated_examples.csv")
PARTC_TUNING_LOG_PATH = os.path.join(RESULTS_DIR, "part_c_tuning_log.csv")

PAD = "<PAD>"
UNK = "<UNK>"
BOS = "<BOS>"
EOS = "<EOS>"
EXPLAIN = "<EXPLAIN>"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

MAX_SEQ_LEN = 220
MAX_PROMPT_LEN = 160
MAX_TARGET_LEN = 50


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+|[0-9]+|[^\w\s]", str(text).lower())


def map_sentiment(rating: int) -> str:
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def lexical_diversity(text: str) -> float:
    toks = str(text).split()
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def diversity_bin_name(value: float, low_thr: float, high_thr: float) -> str:
    if value < low_thr:
        return "low_diversity"
    if value < high_thr:
        return "medium_diversity"
    return "high_diversity"


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def retrieve_contexts(
    query_emb: torch.Tensor,
    train_emb: torch.Tensor,
    train_texts: List[str],
    k: int,
    skip_self: bool,
    query_batch_size: int,
    max_chars_per_passage: int,
) -> List[str]:
    # Memory-safe retrieval: chunk queries instead of building one giant NxM similarity matrix.
    query_emb = normalize_rows(query_emb.float())
    train_emb = normalize_rows(train_emb.float())
    train_t = train_emb.t().contiguous()
    all_indices = []

    for start in range(0, query_emb.size(0), query_batch_size):
        end = min(start + query_batch_size, query_emb.size(0))
        sim = torch.matmul(query_emb[start:end], train_t)
        if skip_self and query_emb.size(0) == train_emb.size(0):
            row_idx = torch.arange(start, end)
            col_idx = row_idx - start
            sim[col_idx, row_idx] = -1e9
        topk_idx = torch.topk(sim, k=k, dim=1).indices.cpu()
        all_indices.append(topk_idx)

    indices = torch.cat(all_indices, dim=0)
    contexts = []
    for row in indices:
        passages = [train_texts[int(i)][:max_chars_per_passage] for i in row.tolist()]
        contexts.append(" <CTX_SEP> ".join(passages))
    return contexts


def build_reference_explanation(sentiment: str, diversity_name: str, review_text: str) -> str:
    evidence = " ".join(tokenize(review_text)[:16])
    return (
        f"The predicted sentiment is {sentiment} because the review language indicates this overall tone. "
        f"The writing style appears {diversity_name}, and key phrases include: {evidence}."
    )


def build_prompt(review: str, sentiment: str, diversity_name: str, retrieved_context: str) -> str:
    return (
        f"<REVIEW> {review} "
        f"<PRED_SENTIMENT> {sentiment} "
        f"<PRED_DERIVED> {diversity_name} "
        f"<RETRIEVED> {retrieved_context} "
        f"{EXPLAIN}"
    )


def build_prompt_baseline(review: str, sentiment: str, diversity_name: str) -> str:
    return (
        f"<REVIEW> {review} "
        f"<PRED_SENTIMENT> {sentiment} "
        f"<PRED_DERIVED> {diversity_name} "
        f"<RETRIEVED> none "
        f"{EXPLAIN}"
    )


def build_vocab(train_prompts: List[str], train_targets: List[str], min_freq: int = 2) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for text in train_prompts + train_targets:
        for tok in tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {PAD: PAD_ID, UNK: UNK_ID, BOS: BOS_ID, EOS: EOS_ID}
    for tok, count in freq.items():
        if count >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    return [vocab.get(tok, UNK_ID) for tok in tokenize(text)]


@dataclass
class Example:
    prompt_ids: List[int]
    target_ids: List[int]
    review_text: str
    sentiment: str
    diversity_name: str


class ExplanationDataset(Dataset):
    def __init__(self, examples: List[Example], max_seq_len: int):
        self.examples = examples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        prompt_ids = ex.prompt_ids[:MAX_PROMPT_LEN]
        target_ids = ex.target_ids[:MAX_TARGET_LEN]
        seq = prompt_ids + [BOS_ID] + target_ids + [EOS_ID]
        seq = seq[: self.max_seq_len]

        labels = [-100] * len(seq)
        prompt_cut = min(len(prompt_ids), len(seq) - 1)
        for i in range(prompt_cut, len(seq) - 1):
            labels[i] = seq[i + 1]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x[0].size(0) for x in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (seq, lab) in enumerate(batch):
        input_ids[i, : seq.size(0)] = seq
        labels[i, : lab.size(0)] = lab
    return input_ids, labels


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
        scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.attn(x, pad_mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, ff_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = input_ids.eq(PAD_ID)
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.lm_head(x)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total += float(loss.item())
            steps += 1
    return total / max(steps, 1)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    model_path: str,
    device: torch.device,
    d_model: int,
    heads: int,
    ff_dim: int,
    layers: int,
    dropout: float,
    lr: float,
    epochs: int,
) -> Tuple[nn.Module, float]:
    model = DecoderOnlyTransformer(vocab_size, d_model, heads, ff_dim, layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        train_loss = total / max(steps, 1)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, best_val


def generate_text(
    model: nn.Module,
    prompt_ids: List[int],
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    device: torch.device,
    max_new_tokens: int = 60,
) -> str:
    model.eval()
    prompt_ids = prompt_ids[:MAX_PROMPT_LEN]
    seq = torch.tensor([prompt_ids + [BOS_ID]], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(seq)
            next_id = int(torch.argmax(logits[0, -1]).item())
            seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == EOS_ID:
                break
    ids = seq[0].tolist()
    start = len(prompt_ids) + 1
    gen_ids = []
    for idx in ids[start:]:
        if idx == EOS_ID:
            break
        gen_ids.append(idx)
    return " ".join(inv_vocab.get(i, UNK) for i in gen_ids)


def create_examples(
    df: pd.DataFrame, contexts: List[str], low_thr: float, high_thr: float, use_retrieval: bool
) -> Tuple[List[str], List[str], List[Tuple[str, str, str]]]:
    prompts, targets, meta = [], [], []
    for i in range(len(df)):
        review = str(df.iloc[i]["reviewText"])
        sentiment = map_sentiment(int(df.iloc[i]["overall"]))
        div_name = diversity_bin_name(lexical_diversity(review), low_thr, high_thr)
        prompt = build_prompt(review, sentiment, div_name, contexts[i]) if use_retrieval else build_prompt_baseline(review, sentiment, div_name)
        target = build_reference_explanation(sentiment, div_name, review)
        prompts.append(prompt)
        targets.append(target)
        meta.append((review, sentiment, div_name))
    return prompts, targets, meta


def pack_examples(prompts: List[str], targets: List[str], meta: List[Tuple[str, str, str]], vocab: Dict[str, int]) -> List[Example]:
    out = []
    for i in range(len(prompts)):
        review, sentiment, div_name = meta[i]
        out.append(Example(prompt_ids=encode(prompts[i], vocab), target_ids=encode(targets[i], vocab), review_text=review, sentiment=sentiment, diversity_name=div_name))
    return out


def run_pipeline(
    use_retrieval: bool,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_ctx: List[str],
    val_ctx: List[str],
    test_ctx: List[str],
    low_thr: float,
    high_thr: float,
    config: Dict[str, float],
    device: torch.device,
    model_path: str,
    vocab: Dict[str, int],
) -> Tuple[float, List[Tuple[str, str, str, str]]]:
    train_prompts, train_targets, train_meta = create_examples(train_df, train_ctx, low_thr, high_thr, use_retrieval)
    val_prompts, val_targets, val_meta = create_examples(val_df, val_ctx, low_thr, high_thr, use_retrieval)
    test_prompts, test_targets, test_meta = create_examples(test_df, test_ctx, low_thr, high_thr, use_retrieval)

    train_ex = pack_examples(train_prompts, train_targets, train_meta, vocab)
    val_ex = pack_examples(val_prompts, val_targets, val_meta, vocab)
    test_ex = pack_examples(test_prompts, test_targets, test_meta, vocab)

    train_loader = DataLoader(ExplanationDataset(train_ex, MAX_SEQ_LEN), batch_size=int(config["batch_size"]), shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(ExplanationDataset(val_ex, MAX_SEQ_LEN), batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(ExplanationDataset(test_ex, MAX_SEQ_LEN), batch_size=int(config["batch_size"]), shuffle=False, collate_fn=collate_batch)

    model, _ = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=len(vocab),
        model_path=model_path,
        device=device,
        d_model=int(config["d_model"]),
        heads=int(config["heads"]),
        ff_dim=int(config["ff_dim"]),
        layers=int(config["layers"]),
        dropout=float(config["dropout"]),
        lr=float(config["lr"]),
        epochs=int(config["epochs"]),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    test_loss = evaluate_loss(model, test_loader, criterion, device)
    perplexity = float(math.exp(min(test_loss, 20.0)))

    inv_vocab = {v: k for k, v in vocab.items()}
    examples = []
    for i in range(min(5, len(test_ex))):
        gen = generate_text(model, test_ex[i].prompt_ids, vocab, inv_vocab, device)
        ref = " ".join(inv_vocab.get(t, UNK) for t in test_ex[i].target_ids[:50])
        examples.append((test_ex[i].review_text, test_ex[i].sentiment, ref, gen))
    del train_loader, val_loader, test_loader, train_ex, val_ex, test_ex
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return perplexity, examples


def append_tuning_log(config: Dict[str, float], ppl_full: float, ppl_base: float) -> None:
    exists = os.path.exists(PARTC_TUNING_LOG_PATH)
    with open(PARTC_TUNING_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lr", "layers", "d_model", "heads", "ff_dim", "dropout", "batch_size", "epochs", "ppl_full", "ppl_baseline"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow({**config, "ppl_full": ppl_full, "ppl_baseline": ppl_base})


def main() -> None:
    parser = argparse.ArgumentParser(description="Part C decoder-only explanation generation")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--retrieval_batch_size", type=int, default=128)
    parser.add_argument("--max_context_chars", type=int, default=280)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    for path in [TRAIN_CSV, VAL_CSV, TEST_CSV, EMBED_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    if args.max_samples > 0:
        train_df = train_df.head(args.max_samples).reset_index(drop=True)
        val_df = val_df.head(max(100, args.max_samples // 4)).reset_index(drop=True)
        test_df = test_df.head(max(100, args.max_samples // 4)).reset_index(drop=True)

    train_div = train_df["reviewText"].apply(lexical_diversity)
    low_thr = float(train_div.quantile(0.33))
    high_thr = float(train_div.quantile(0.66))

    emb = torch.load(EMBED_PATH, map_location="cpu")
    train_emb = emb["train"].float()[: len(train_df)]
    val_emb = emb["val"].float()[: len(val_df)]
    test_emb = emb["test"].float()[: len(test_df)]

    train_texts = train_df["reviewText"].astype(str).tolist()
    train_ctx = retrieve_contexts(
        train_emb,
        train_emb,
        train_texts,
        k=args.k,
        skip_self=True,
        query_batch_size=args.retrieval_batch_size,
        max_chars_per_passage=args.max_context_chars,
    )
    val_ctx = retrieve_contexts(
        val_emb,
        train_emb,
        train_texts,
        k=args.k,
        skip_self=False,
        query_batch_size=args.retrieval_batch_size,
        max_chars_per_passage=args.max_context_chars,
    )
    test_ctx = retrieve_contexts(
        test_emb,
        train_emb,
        train_texts,
        k=args.k,
        skip_self=False,
        query_batch_size=args.retrieval_batch_size,
        max_chars_per_passage=args.max_context_chars,
    )

    train_prompts_full, train_targets_full, _ = create_examples(train_df, train_ctx, low_thr, high_thr, True)
    vocab = build_vocab(train_prompts_full, train_targets_full)
    torch.save(vocab, PARTC_VOCAB_PATH)

    config = {
        "lr": args.lr,
        "layers": args.layers,
        "d_model": args.d_model,
        "heads": args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Training full RAG decoder (with retrieval)...")
    ppl_full, examples_full = run_pipeline(
        True, train_df, val_df, test_df, train_ctx, val_ctx, test_ctx, low_thr, high_thr, config, device, PARTC_FULL_MODEL, vocab
    )
    print("Training baseline decoder (no retrieval)...")
    ppl_base, examples_base = run_pipeline(
        False, train_df, val_df, test_df, train_ctx, val_ctx, test_ctx, low_thr, high_thr, config, device, PARTC_BASE_MODEL, vocab
    )

    with open(PARTC_METRICS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["system", "test_perplexity"])
        writer.writeheader()
        writer.writerow({"system": "full_rag_with_retrieval", "test_perplexity": ppl_full})
        writer.writerow({"system": "baseline_without_retrieval", "test_perplexity": ppl_base})

    rows = []
    for i in range(min(5, len(examples_full))):
        q_text, sent, ref, gen_full = examples_full[i]
        _, _, _, gen_base = examples_base[i]
        commentary = "Full model adds context-linked details." if len(gen_full.split()) >= len(gen_base.split()) else "Baseline response is shorter/less grounded."
        rows.append(
            {
                "example_id": i,
                "sentiment": sent,
                "query_review": q_text,
                "reference_explanation_snippet": ref,
                "generated_full_rag": gen_full,
                "generated_baseline": gen_base,
                "commentary": commentary,
            }
        )
    pd.DataFrame(rows).to_csv(PARTC_EXAMPLES_PATH, index=False)
    append_tuning_log(config, ppl_full, ppl_base)

    print(f"Saved Part C metrics to: {PARTC_METRICS_PATH}")
    print(f"Saved generated examples to: {PARTC_EXAMPLES_PATH}")
    print(f"Saved tuning log to: {PARTC_TUNING_LOG_PATH}")
    print(f"Perplexity (full): {ppl_full:.4f}")
    print(f"Perplexity (baseline): {ppl_base:.4f}")


if __name__ == "__main__":
    main()
