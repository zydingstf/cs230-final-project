import os
import sys
import time
import logging
import math

from tqdm.auto import tqdm

# Path of this file: project_root/models/transformer.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split
from src.evaluation_metrics import topk_accuracy
import re
from collections import Counter

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
logger.info("Loading data from data/cleaned.csv")
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "cleaned.csv"))
df = df[["text", "label_id"]]
df["text"] = df["text"].fillna("")

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)

logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# ------------------------------------------------------------------
# Tokenizer & vocab
# ------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+|[.,!?;]")

def basic_tokenizer(text: str):
    text = str(text).lower()
    return TOKEN_PATTERN.findall(text)

def build_vocab_from_iterator(texts, min_freq=2, specials=None):
    if specials is None:
        specials = ["<PAD>", "<UNK>"]

    counter = Counter()
    for t in texts:
        tokens = basic_tokenizer(t)
        counter.update(tokens)

    stoi = {}
    for sp in specials:
        if sp not in stoi:
            stoi[sp] = len(stoi)

    for tok, freq in counter.items():
        if freq >= min_freq and tok not in stoi:
            stoi[tok] = len(stoi)

    itos = {i: s for s, i in stoi.items()}
    pad_idx = stoi["<PAD>"]
    unk_idx = stoi["<UNK>"]

    def encode(tokens):
        return [stoi.get(tok, unk_idx) for tok in tokens]

    vocab = {
        "stoi": stoi,
        "itos": itos,
        "pad_idx": pad_idx,
        "unk_idx": unk_idx,
        "encode": encode,
    }
    return vocab

logger.info("Building vocabulary...")
vocab = build_vocab_from_iterator(
    train_df["text"].tolist(),
    min_freq=2,
    specials=["<PAD>", "<UNK>"],
)

pad_idx = vocab["pad_idx"]
vocab_size = len(vocab["stoi"])
num_classes = df["label_id"].nunique()
logger.info(f"Vocab size: {vocab_size}, Num classes: {num_classes}")

def text_to_indices(text, vocab):
    if pd.isna(text):
        text = ""
    tokens = basic_tokenizer(text)
    if len(tokens) == 0:
        tokens = ["<UNK>"]
    ids = vocab["encode"](tokens)
    return torch.tensor(ids, dtype=torch.long)

# ------------------------------------------------------------------
# Dataset & loaders
# ------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, df_, vocab):
        self.texts = df_["text"].tolist()
        self.labels = df_["label_id"].tolist()
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = text_to_indices(self.texts[idx], self.vocab)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    padded_seqs = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=pad_idx,
    )
    labels = torch.stack(labels)
    return padded_seqs, lengths, labels

train_dataset = TextDataset(train_df, vocab)
val_dataset   = TextDataset(val_df, vocab)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
)

# ------------------------------------------------------------------
# Transformer model
# ------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        pad_idx: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # [batch, seq_len, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len]
        lengths: [batch]
        """
        pad_mask = (x == self.pad_idx)

        emb = self.embedding(x)
        emb = self.pos_encoder(emb)

        encoded = self.transformer_encoder(
            emb, src_key_padding_mask=pad_mask
        )

        lengths = lengths.clamp(min=1).unsqueeze(1)

        mask = (~pad_mask).unsqueeze(-1)
        encoded_masked = encoded * mask
        summed = encoded_masked.sum(dim=1)
        pooled = summed / lengths

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        return logits

# ------------------------------------------------------------------
# Training setup
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = TransformerClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    dim_feedforward=256,
    num_classes=num_classes,
    pad_idx=pad_idx,
    dropout=0.1,
    max_seq_len=256,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
num_epochs = 20
logger.info(f"Starting training for {num_epochs} epochs")

overall_start = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()

    # ---------- Training ----------
    model.train()
    total_loss = 0.0
    total_train_examples = 0
    train_scores = []
    train_labels = []

    for batch_x, batch_lengths, batch_y in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", leave=False
    ):
        batch_x = batch_x.to(device)
        batch_lengths = batch_lengths.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x, batch_lengths)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_train_examples += batch_size

        train_scores.append(logits.detach().cpu().numpy())
        train_labels.append(batch_y.detach().cpu().numpy())

    avg_loss = total_loss / total_train_examples
    y_scores_train = np.concatenate(train_scores, axis=0)
    y_true_train   = np.concatenate(train_labels, axis=0)
    train_top1 = topk_accuracy(y_true_train, y_scores_train, k=1)
    train_top3 = topk_accuracy(y_true_train, y_scores_train, k=3)

    # ---------- Validation ----------
    model.eval()
    val_scores = []
    val_labels = []

    for batch_x, batch_lengths, batch_y in tqdm(
        val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]", leave=False
    ):
        batch_x = batch_x.to(device)
        batch_lengths = batch_lengths.to(device)
        batch_y = batch_y.to(device)

        with torch.inference_mode():
            logits = model(batch_x, batch_lengths)

        val_scores.append(logits.detach().cpu().numpy())
        val_labels.append(batch_y.detach().cpu().numpy())

    y_scores_val = np.concatenate(val_scores, axis=0)
    y_true_val   = np.concatenate(val_labels, axis=0)
    val_top1 = topk_accuracy(y_true_val, y_scores_val, k=1)
    val_top3 = topk_accuracy(y_true_val, y_scores_val, k=3)

    epoch_time = time.time() - epoch_start

    logger.info(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"train_loss={avg_loss:.4f} | "
        f"train_top1={train_top1:.4f} | train_top3={train_top3:.4f} | "
        f"val_top1={val_top1:.4f} | val_top3={val_top3:.4f} | "
        f"epoch_time={epoch_time:.1f}s"
    )

overall_time = time.time() - overall_start
logger.info(f"Training finished in {overall_time:.1f} seconds")
