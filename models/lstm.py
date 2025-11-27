import os
import sys
import time
import logging

# Progress bar
from tqdm.auto import tqdm

# Path of this file: project_root/models/lstm.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # go up to project_root

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

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from sklearn.model_selection import train_test_split
from src.evaluation_metrics import topk_accuracy

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
logger.info("Loading data from data/cleaned.csv")
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "cleaned.csv"))
df = df[["text", "label_id"]]

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)

logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

# ------------------------------------------------------------------
# Tokenizer & vocab
# ------------------------------------------------------------------
tokenizer = get_tokenizer("basic_english")

def yield_tokens(df_):
    for text in df_["text"]:
        yield tokenizer(text)

logger.info("Building vocabulary...")
vocab = build_vocab_from_iterator(
    yield_tokens(train_df),
    min_freq=2,
    specials=["<PAD>", "<UNK>"],
)
vocab.set_default_index(vocab["<UNK>"])

pad_idx = vocab["<PAD>"]
vocab_size = len(vocab)
num_classes = df["label_id"].nunique()
logger.info(f"Vocab size: {vocab_size}, Num classes: {num_classes}")

def text_to_indices(text, tokenizer, vocab):
    # handle NaNs just in case
    if pd.isna(text):
        text = ""

    tokens = tokenizer(text)

    # if token list is empty, fall back to a single <UNK> token
    if len(tokens) == 0:
        tokens = ["<UNK>"]

    ids = vocab(tokens)
    return torch.tensor(ids, dtype=torch.long)

# ------------------------------------------------------------------
# Dataset & DataLoaders
# ------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, df_, tokenizer, vocab):
        self.texts = df_["text"].tolist()
        self.labels = df_["label_id"].tolist()
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = text_to_indices(self.texts[idx], self.tokenizer, self.vocab)
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

train_dataset = TextDataset(train_df, tokenizer, vocab)
val_dataset   = TextDataset(val_df, tokenizer, vocab)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_classes,
        num_layers=1,
        bidirectional=True,
        dropout=0.5,
        pad_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            h = torch.cat((h_forward, h_backward), dim=1)
        else:
            h = h_n[-1, :, :]

        h = self.dropout(h)
        logits = self.fc(h)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = LSTMClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    hidden_dim=128,
    num_classes=num_classes,
    num_layers=1,
    bidirectional=True,
    dropout=0.5,
    pad_idx=pad_idx,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ------------------------------------------------------------------
# Training loop with progress + timing
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

    # tqdm progress bar over training batches
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
