import os
import sys
import time
import logging

from tqdm.auto import tqdm

# Path of this file: project_root/models/transformer_hf.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging
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

from sklearn.model_selection import train_test_split
from src.evaluation_metrics import topk_accuracy

from transformers import AutoTokenizer, AutoModel

# Model config
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 64
LR = 2e-5
NUM_EPOCHS = 5

# Checkpoint setup
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints") 
ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "analysis")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

BEST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR, f"{MODEL_NAME}_best.pt"
)


# Data loading
logger.info("Loading data from data/cleaned.csv")
df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "cleaned.csv"))
df = df[["text", "label_id"]]
df["text"] = df["text"].fillna("")

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_id"]
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

num_classes = df["label_id"].nunique()
logger.info(f"Num classes: {num_classes}")

# Tokenizer
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset
class TextDataset(Dataset):
    def __init__(self, df_):
        self.texts = df_["text"].tolist()
        self.labels = df_["label_id"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        return text, label

def collate_fn(batch):
    """
    batch: list of (text, label)
    We tokenize inside collate_fn so we can use HF's efficient batch tokenization.
    """
    texts, labels = zip(*batch)

    # Batch encode with padding & truncation
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    labels = torch.tensor(labels, dtype=torch.long)
    return encoded, labels

train_dataset = TextDataset(train_df)
val_dataset = TextDataset(val_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE_VAL,
    shuffle=False,
    collate_fn=collate_fn,
)

# Model
class HFEmojiClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # use first token ([CLS]-like) as sentence representation
        last_hidden_state = outputs.last_hidden_state 
        cls_rep = last_hidden_state[:, 0, :]

        cls_rep = self.dropout(cls_rep)
        logits = self.classifier(cls_rep)
        return logits

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = HFEmojiClassifier(
    model_name=MODEL_NAME,
    num_classes=num_classes,
    dropout=0.2,
).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

total_train_steps = NUM_EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_train_steps
)

# Resume from checkpoint
best_val_top1 = 0.0
start_epoch = 0
if os.path.exists(BEST_CHECKPOINT_PATH):
    logger.info(f"Found checkpoint at {BEST_CHECKPOINT_PATH}, loading...")
    ckpt = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"]
    best_val_top1 = ckpt.get("val_top1", 0.0)
    logger.info(
        f"Resuming from epoch {start_epoch} with best_val_top1={best_val_top1:.4f}"
    )


# Training loop
logger.info(f"Starting training for {NUM_EPOCHS} epochs")

overall_start = time.time()
global_step = 0

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()

    model.train()
    total_loss = 0.0
    total_train_examples = 0
    train_scores = []
    train_labels = []

    for batch_inputs, batch_y in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [train]", leave=False
    ):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
        )
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1

        batch_size = batch_y.size(0)
        total_loss += loss.item() * batch_size
        total_train_examples += batch_size

        train_scores.append(logits.detach().cpu().numpy())
        train_labels.append(batch_y.detach().cpu().numpy())

    avg_loss = total_loss / total_train_examples
    y_scores_train = np.concatenate(train_scores, axis=0)
    y_true_train   = np.concatenate(train_labels, axis=0)
    train_top1 = topk_accuracy(y_true_train, y_scores_train, k=1)
    train_top3 = topk_accuracy(y_true_train, y_scores_train, k=3)

    # Validatiaon
    model.eval()
    val_scores = []
    val_labels = []

    for batch_inputs, batch_y in tqdm(
        val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [val]", leave=False
    ):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_y = batch_y.to(device)

        with torch.inference_mode():
            logits = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
            )

        val_scores.append(logits.detach().cpu().numpy())
        val_labels.append(batch_y.detach().cpu().numpy())

    y_scores_val = np.concatenate(val_scores, axis=0)
    y_true_val = np.concatenate(val_labels, axis=0)
    val_top1 = topk_accuracy(y_true_val, y_scores_val, k=1)
    val_top3 = topk_accuracy(y_true_val, y_scores_val, k=3)

    epoch_time = time.time() - epoch_start

    logger.info(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"train_loss={avg_loss:.4f} | "
        f"train_top1={train_top1:.4f} | train_top3={train_top3:.4f} | "
        f"val_top1={val_top1:.4f} | val_top3={val_top3:.4f} | "
        f"epoch_time={epoch_time:.1f}s"
    )

    # Save best checkpoint
    if val_top1 > best_val_top1:
        best_val_top1 = val_top1
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_top1": val_top1,
            "val_top3": val_top3,
        }
        torch.save(ckpt, BEST_CHECKPOINT_PATH)
        logger.info(
            f"New best model saved to {BEST_CHECKPOINT_PATH} (val_top1={val_top1:.4f})"
        )

    # Misclassification analysis for the final epoch
    if epoch == NUM_EPOCHS - 1:
        y_pred_val = y_scores_val.argmax(axis=1)

        top3_idx = np.argsort(-y_scores_val, axis=1)[:, :3]

        val_analysis_df = val_df.copy().reset_index(drop=True)
        val_analysis_df["true_label"] = y_true_val
        val_analysis_df["pred_label"] = y_pred_val
        val_analysis_df["top3_labels"] = top3_idx.tolist()

        misclassified_df = val_analysis_df[
            val_analysis_df["true_label"] != val_analysis_df["pred_label"]
        ]

        val_preds_path = os.path.join(
            ANALYSIS_DIR, f"val_predictions_epoch{epoch+1}.csv"
        )
        miscls_path = os.path.join(
            ANALYSIS_DIR, f"val_misclassified_epoch{epoch+1}.csv"
        )

        val_analysis_df.to_csv(val_preds_path, index=False)
        misclassified_df.to_csv(miscls_path, index=False)

        logger.info(
            f"[FINAL EPOCH] Saved validation predictions to {val_preds_path} "
            f"and misclassified examples to {miscls_path} "
            f"(num_misclassified={len(misclassified_df)})"
        )


overall_time = time.time() - overall_start
logger.info(f"Training finished in {overall_time:.1f} seconds")
