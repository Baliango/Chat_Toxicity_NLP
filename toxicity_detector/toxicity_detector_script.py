# pip install --upgrade transformers
# pip install --upgrade torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

model_path = "../pipeline_assets/toxicibert_final-20240313T210417Z-001/toxicibert_final/"
tokenizer_path = "../pipeline_assets/toxicibert_tokenizer_final-20240313T210419Z-001/toxicibert_tokenizer_final/"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def toxicity_detector(sentence):  
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probabilities = F.softmax(outputs.logits, dim=1)
    prediction = "Toxic" if probabilities[0][1] > 0.5 else "Non-toxic"
    return prediction

result = toxicity_detector("You are a toxic person, shame on you")
print(result)