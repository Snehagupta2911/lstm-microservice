import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# 1. Recreate your LSTM class
# -----------------------------
PAD_IDX = 1      # same as your notebook
VOCAB_SIZE = 25002  # same vocab size used during training

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        emb = self.embedding(x)
        output, (hidden, cell) = self.lstm(emb)
        hidden_forward = hidden[-2,:,:]
        hidden_backward = hidden[-1,:,:]
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        out = self.dropout(final_hidden)
        return self.fc(out)

# -----------------------------
# 2. Load trained weights
# -----------------------------
model = SentimentLSTM()
model.load_state_dict(torch.load("sentiment_lstm_model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# 3. FastAPI setup
# -----------------------------
app = FastAPI(
    title="LSTM Sentiment Analysis API",
    description="A microservice exposing the LSTM model from Assignment 4.",
    version="1.0"
)

# -----------------------------
# 4. Input format
# -----------------------------
class TextInput(BaseModel):
    text: str

# -----------------------------
# 5. Preprocessing function
# (very simplified)
# -----------------------------
import re
import torch

def simple_tokenizer(text):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    # Convert tokens → fake IDs (for demonstration)
    ids = [min(abs(hash(t)) % VOCAB_SIZE, VOCAB_SIZE - 1) for t in tokens]
    return torch.tensor(ids).unsqueeze(0)

# -----------------------------
# 6. Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(input: TextInput):
    tokens = simple_tokenizer(input.text)
    with torch.no_grad():
        logits = model(tokens)
        prediction = torch.argmax(logits, dim=1).item()

    label = "positive" if prediction == 1 else "negative"

    return {
        "input_text": input.text,
        "prediction_label": label,
        "prediction_raw": prediction
    }

@app.get("/")
def home():
    return {"message": "LSTM Sentiment API is running!"}
