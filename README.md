# Emoji Prediction from Tweets

This project builds machine learning models to predict the most likely emoji given a tweet. Emoji usage is often context-dependent and subjective, making this a challenging multi-class classification problem. We experiment with a range of models from classical baselines to deep pretrained Transformers to understand how linguistic representation influences prediction performance.

---

## Models Implemented

| Model | Description |
|---|---|
| Logistic Regression | TF-IDF baseline; fast but limited representation power |
| MLP | Adds nonlinearity over TF-IDF vectors; prone to overfitting |
| LSTM | Sequential modeling of text; better semantic capture |
| Custom Transformer | Attention-based architecture for contextual reasoning |
| DistilBERT (Fine-Tuned) | Best performing model; strong pretrained embeddings |
| DistilBERT + LoRA | Parameter-efficient fine-tuning with competitive accuracy |
