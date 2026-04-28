import csv
import math
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "Dataset")
DATASET_PT = os.path.join(DATA_DIR, "dataset.pt")
TRAIN_CSV = os.path.join(DATA_DIR, "dataset_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "dataset_val.csv")
TEST_CSV = os.path.join(DATA_DIR, "dataset_test.csv")

TEXT_COL = "reviewText"
RATING_COL = "overall"
PAD_IDX = 0

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("EPOCHS", "10"))
LR = float(os.getenv("LR", "1e-3"))
ALPHA_SENTIMENT = 1.5
ALPHA_DERIVED = 0.5
USE_UNCERTAINTY_WEIGHTING = os.getenv("USE_UNCERTAINTY_WEIGHTING", "1") == "1"
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "3"))
GRAD_CLIP_NORM = float(os.getenv("GRAD_CLIP_NORM", "1.0"))

EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 2
WEIGHT_DECAY = 5e-4
DROPOUT = 0.3

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "multitask_transformer.pt")
CURVES_CSV_PATH = os.path.join(RESULTS_DIR, "learning_curves.csv")
CURVES_PNG_PATH = os.path.join(RESULTS_DIR, "learning_curves.png")
EMBED_PATH = os.path.join(RESULTS_DIR, "review_embeddings.pt")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def map_sentiment(rating):
    if rating <= 2:
        return 0  # Negative
    if rating == 3:
        return 1  # Neutral
    return 2  # Positive


def resolve_csv_path(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(
        f"Could not find {filename} in {DATA_DIR}."
    )


def count_words(text):
    return len(str(text).split())


def lexical_diversity(text):
    tokens = str(text).split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def macro_f1(y_true, y_pred, num_classes):
    f1_scores = []
    for class_id in range(num_classes):
        tp = ((y_true == class_id) & (y_pred == class_id)).sum().item()
        fp = ((y_true != class_id) & (y_pred == class_id)).sum().item()
        fn = ((y_true == class_id) & (y_pred != class_id)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)


@dataclass
class SplitTargets:
    sentiment: torch.Tensor
    derived_label: torch.Tensor


class ReviewDataset(Dataset):
    def __init__(self, input_ids, sentiment_labels, derived_labels):
        self.input_ids = input_ids
        self.sentiment_labels = sentiment_labels
        self.derived_labels = derived_labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.sentiment_labels[idx],
            self.derived_labels[idx],
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
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

    def forward(self, x, pad_mask):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # pad_mask: (batch, seq) True where padded
        scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
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
        # Residual gates reduce direct shortcut dominance early in training.
        self.attn_gate = nn.Parameter(torch.tensor(0.5))
        self.ffn_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, pad_mask):
        attn_out = self.self_attn(x, pad_mask)
        x = self.norm1(x + self.attn_gate * self.dropout1(attn_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.ffn_gate * self.dropout2(ff_out))
        return x


class MultiTaskTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(EMBED_DIM)
        self.encoder_layers = nn.ModuleList(
            [EncoderBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT) for _ in range(NUM_LAYERS)]
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.pool_query = nn.Linear(EMBED_DIM, 1)
        self.sent_adapter = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.der_adapter = nn.Linear(EMBED_DIM, EMBED_DIM)
        self.sentiment_head = nn.Linear(EMBED_DIM, 3)
        self.derived_head = nn.Linear(EMBED_DIM, 3)
        self.log_var_sent = nn.Parameter(torch.zeros(1))
        self.log_var_der = nn.Parameter(torch.zeros(1))

    def attention_pool(self, token_repr, mask):
        scores = self.pool_query(token_repr).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return (token_repr * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, input_ids):
        pad_mask = input_ids.eq(PAD_IDX)
        token_mask = (~pad_mask).float()

        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, pad_mask)
        pooled = self.attention_pool(x, token_mask)
        pooled = self.dropout(pooled)

        sent_feat = torch.relu(self.sent_adapter(pooled))
        der_feat = torch.relu(self.der_adapter(pooled))
        sentiment_logits = self.sentiment_head(sent_feat)
        derived_logits = self.derived_head(der_feat)
        return sentiment_logits, derived_logits, pooled


def map_derived_bin(diversity_value, low_thr, high_thr):
    if diversity_value < low_thr:
        return 0  # low
    if diversity_value < high_thr:
        return 1  # medium
    return 2  # high


def build_targets(csv_path, low_thr, high_thr):
    df = pd.read_csv(csv_path)
    sentiment = torch.tensor(df[RATING_COL].apply(map_sentiment).tolist(), dtype=torch.long)
    diversity = df[TEXT_COL].apply(lexical_diversity)
    derived = torch.tensor(
        diversity.apply(lambda x: map_derived_bin(x, low_thr, high_thr)).tolist(),
        dtype=torch.long,
    )
    return SplitTargets(sentiment=sentiment, derived_label=derived), diversity.tolist()


def multitask_loss(model, loss_sentiment, loss_derived):
    if USE_UNCERTAINTY_WEIGHTING:
        return (
            torch.exp(-model.log_var_sent) * loss_sentiment
            + model.log_var_sent
            + torch.exp(-model.log_var_der) * loss_derived
            + model.log_var_der
        ).mean()
    return ALPHA_SENTIMENT * loss_sentiment + ALPHA_DERIVED * loss_derived


def evaluate(model, loader, sentiment_loss_fn, derived_loss_fn):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_sentiment_true = []
    all_sentiment_pred = []
    all_derived_true = []
    all_derived_pred = []

    with torch.no_grad():
        for input_ids, sentiment_y, derived_y in loader:
            input_ids = input_ids.to(DEVICE)
            sentiment_y = sentiment_y.to(DEVICE)
            derived_y = derived_y.to(DEVICE)

            sentiment_logits, derived_logits, _ = model(input_ids)

            loss_sentiment = sentiment_loss_fn(sentiment_logits, sentiment_y)
            loss_derived = derived_loss_fn(derived_logits, derived_y)
            loss = multitask_loss(model, loss_sentiment, loss_derived)

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            sentiment_pred = sentiment_logits.argmax(dim=1)
            all_sentiment_true.append(sentiment_y.cpu())
            all_sentiment_pred.append(sentiment_pred.cpu())
            derived_pred = derived_logits.argmax(dim=1)

            all_derived_true.append(derived_y.cpu())
            all_derived_pred.append(derived_pred.cpu())

    sentiment_true = torch.cat(all_sentiment_true)
    sentiment_pred = torch.cat(all_sentiment_pred)
    derived_true = torch.cat(all_derived_true)
    derived_pred = torch.cat(all_derived_pred)

    sentiment_acc = (sentiment_true == sentiment_pred).float().mean().item()
    sentiment_f1 = macro_f1(sentiment_true, sentiment_pred, num_classes=3)
    derived_acc = (derived_true == derived_pred).float().mean().item()
    derived_f1 = macro_f1(derived_true, derived_pred, num_classes=3)

    return {
        "loss": total_loss / max(total_samples, 1),
        "sentiment_acc": sentiment_acc,
        "sentiment_macro_f1": sentiment_f1,
        "derived_acc": derived_acc,
        "derived_macro_f1": derived_f1,
    }


def save_curves(history):
    with open(CURVES_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_sentiment_acc = [row["val_sentiment_acc"] for row in history]
    val_sentiment_f1 = [row["val_sentiment_macro_f1"] for row in history]
    val_derived_acc = [row["val_derived_acc"] for row in history]
    val_derived_f1 = [row["val_derived_macro_f1"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curves")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_sentiment_acc, label="val_sentiment_acc")
    plt.plot(epochs, val_sentiment_f1, label="val_sentiment_macro_f1")
    plt.plot(epochs, val_derived_acc, label="val_derived_acc")
    plt.plot(epochs, val_derived_f1, label="val_derived_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(CURVES_PNG_PATH, dpi=200)
    plt.close()


def extract_embeddings(model, input_ids, batch_size=BATCH_SIZE):
    model.eval()
    vectors = []
    loader = DataLoader(input_ids, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_ids in loader:
            batch_ids = batch_ids.to(DEVICE)
            _, _, pooled = model(batch_ids)
            vectors.append(pooled.cpu())
    return torch.cat(vectors, dim=0)


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATASET_PT):
        raise FileNotFoundError(
            f"{DATASET_PT} not found. Run preprocessing and place output in Dataset/dataset.pt."
        )

    tensor_data = torch.load(DATASET_PT)
    train_input_ids = tensor_data["train"].long()
    val_input_ids = tensor_data["val"].long()
    test_input_ids = tensor_data["test"].long()
    vocab = tensor_data["vocab"]

    print("Derived task: predict lexical diversity class (low/medium/high)")
    train_csv_path = resolve_csv_path("dataset_train.csv")
    val_csv_path = resolve_csv_path("dataset_val.csv")
    test_csv_path = resolve_csv_path("dataset_test.csv")

    train_df = pd.read_csv(train_csv_path)
    train_diversity = train_df[TEXT_COL].apply(lexical_diversity)
    low_thr = float(train_diversity.quantile(0.33))
    high_thr = float(train_diversity.quantile(0.66))
    print(f"Derived bins (from train set): low < {low_thr:.3f}, medium < {high_thr:.3f}, high >= {high_thr:.3f}")

    train_targets, _ = build_targets(train_csv_path, low_thr, high_thr)
    val_targets, _ = build_targets(val_csv_path, low_thr, high_thr)
    test_targets, _ = build_targets(test_csv_path, low_thr, high_thr)

    train_dataset = ReviewDataset(train_input_ids, train_targets.sentiment, train_targets.derived_label)
    val_dataset = ReviewDataset(val_input_ids, val_targets.sentiment, val_targets.derived_label)
    test_dataset = ReviewDataset(test_input_ids, test_targets.sentiment, test_targets.derived_label)

    model = MultiTaskTransformer(vocab_size=len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    class_counts = torch.bincount(train_targets.sentiment, minlength=3).float()
    inv_freq = class_counts.sum() / (3.0 * class_counts.clamp(min=1.0))
    class_weights = inv_freq.sqrt()
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(DEVICE)
    sentiment_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    derived_counts = torch.bincount(train_targets.derived_label, minlength=3).float()
    derived_inv_freq = derived_counts.sum() / (3.0 * derived_counts.clamp(min=1.0))
    derived_weights = derived_inv_freq.sqrt()
    derived_weights = derived_weights / derived_weights.mean()
    derived_weights = derived_weights.to(DEVICE)
    derived_loss_fn = nn.CrossEntropyLoss(weight=derived_weights)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-5)

    best_val_f1 = -1.0
    no_improve_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        for input_ids, sentiment_y, derived_y in train_loader:
            input_ids = input_ids.to(DEVICE)
            sentiment_y = sentiment_y.to(DEVICE)
            derived_y = derived_y.to(DEVICE)

            sentiment_logits, derived_logits, _ = model(input_ids)
            loss_sentiment = sentiment_loss_fn(sentiment_logits, sentiment_y)
            loss_derived = derived_loss_fn(derived_logits, derived_y)
            loss = multitask_loss(model, loss_sentiment, loss_derived)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            batch_size = input_ids.size(0)
            total_train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        train_loss = total_train_loss / max(total_train_samples, 1)
        val_metrics = evaluate(model, val_loader, sentiment_loss_fn, derived_loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_sentiment_acc": val_metrics["sentiment_acc"],
                "val_sentiment_macro_f1": val_metrics["sentiment_macro_f1"],
                "val_derived_acc": val_metrics["derived_acc"],
                "val_derived_macro_f1": val_metrics["derived_macro_f1"],
            }
        )

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_sent_acc={val_metrics['sentiment_acc']:.4f} | "
            f"val_sent_f1={val_metrics['sentiment_macro_f1']:.4f} | "
            f"val_der_acc={val_metrics['derived_acc']:.4f} | "
            f"val_der_f1={val_metrics['derived_macro_f1']:.4f}"
        )

        scheduler.step(val_metrics["loss"])

        if val_metrics["sentiment_macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["sentiment_macro_f1"]
            no_improve_epochs = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_metrics = evaluate(model, test_loader, sentiment_loss_fn, derived_loss_fn)
    print("\nTest metrics:")
    print(f"Sentiment accuracy: {test_metrics['sentiment_acc']:.4f}")
    print(f"Sentiment macro-F1: {test_metrics['sentiment_macro_f1']:.4f}")
    print(f"Derived-task accuracy: {test_metrics['derived_acc']:.4f}")
    print(f"Derived-task macro-F1: {test_metrics['derived_macro_f1']:.4f}")
    print(f"Combined test loss: {test_metrics['loss']:.4f}")

    save_curves(history)

    embeddings = {
        "train": extract_embeddings(model, train_input_ids),
        "val": extract_embeddings(model, val_input_ids),
        "test": extract_embeddings(model, test_input_ids),
    }
    torch.save(embeddings, EMBED_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved learning curves to: {CURVES_CSV_PATH} and {CURVES_PNG_PATH}")
    print(f"Saved review embeddings to: {EMBED_PATH}")


if __name__ == "__main__":
    main()
