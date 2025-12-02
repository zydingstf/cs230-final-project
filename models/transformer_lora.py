import os
import sys
import time
import logging

from tqdm.auto import tqdm

# Path of this file: project_root/models/transformer_hf_lora.py
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

from sklearn.model_selection import train_test_split
from src.evaluation_metrics import topk_accuracy

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"   # bert-base-uncased, roberta-base
MAX_LEN = 128
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 64
LR = 2e-5
NUM_EPOCHS = 5 
WEIGHT_DECAY = 0.01

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

num_classes = df["label_id"].nunique()
logger.info(f"Num classes: {num_classes}")

# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------
logger.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ------------------------------------------------------------------
# Dataset & DataLoaders
# ------------------------------------------------------------------
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
    Tokenize inside collate_fn for efficient batched encoding.
    """
    texts, labels = zip(*batch)

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
val_dataset   = TextDataset(val_df)

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

# ------------------------------------------------------------------
# Model: Pretrained Transformer + LoRA
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

logger.info(f"Loading base model: {MODEL_NAME}")
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
)

# LoRA config for DistilBERT:
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,                   # rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # DistilBERT-specific
)

logger.info("Wrapping model with LoRA...")
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

total_train_steps = NUM_EPOCHS * len(train_loader) if len(train_loader) > 0 else 1
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=total_train_steps,
)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
logger.info(f"Starting training for {NUM_EPOCHS} epochs with LoRA")

overall_start = time.time()
global_step = 0

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # ---------- Training ----------
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

        outputs = model(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            labels=batch_y,
        )

        logits = outputs.logits
        loss = outputs.loss

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

    # ---------- Validation ----------
    model.eval()
    val_scores = []
    val_labels = []

    for batch_inputs, batch_y in tqdm(
        val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [val]", leave=False
    ):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        batch_y = batch_y.to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                labels=batch_y,
            )
            logits = outputs.logits

        val_scores.append(logits.detach().cpu().numpy())
        val_labels.append(batch_y.detach().cpu().numpy())

    y_scores_val = np.concatenate(val_scores, axis=0)
    y_true_val   = np.concatenate(val_labels, axis=0)
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

overall_time = time.time() - overall_start
logger.info(f"Training finished in {overall_time:.1f} seconds")
