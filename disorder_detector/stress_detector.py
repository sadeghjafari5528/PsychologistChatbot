"""
    First the transformers library needs to be installed using the below command:
        ```!pip install -q transformers```
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def load_stress_detector_model_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("AylinNaebzadeh/XLM-RoBERTa-FineTuned-With-Dreaddit")
    model = AutoModelForSequenceClassification.from_pretrained("AylinNaebzadeh/XLM-RoBERTa-FineTuned-With-Dreaddit")
    return tokenizer, model


def check_for_stress_in_text(input_text):
    stress_model_detc, stress_tokenizer_detc = load_stress_detector_model_tokenizer()
    inputs = stress_tokenizer_detc(input_text, return_tensors="pt")
    with torch.no_grad():
        logits = stress_model_detc(**inputs).logits
    predicted_class_id = logits.argmax().item()
    output_label = stress_model_detc.config.id2label[predicted_class_id]
    if output_label == "LABEL_0":
        return "Not Stressed"
    return "Stressed"