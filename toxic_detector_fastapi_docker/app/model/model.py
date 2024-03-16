from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

tokenizer = AutoTokenizer.from_pretrained(BASE_DIR / "pipeline_assets/toxicibert_tokenizer_0.1.0/")
model = AutoModelForSequenceClassification.from_pretrained(BASE_DIR / "pipeline_assets/toxicibert_0.1.0/")

def detect_toxicity(sentence):  
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1)
    prediction = "Toxic" if probabilities[0][1] > 0.5 else "Non-toxic"
    return prediction