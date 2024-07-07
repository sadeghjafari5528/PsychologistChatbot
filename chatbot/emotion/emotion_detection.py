from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_dict = {
    0: 'OTHER',
    1: 'HAPPY',
    2: 'SURPRISE',
    3: 'FEAR',
    4: 'HATE',
    5: 'ANGRY',
    6: 'SAD',
}


def load_emotion_detector_model_tokenizer(model_name="AmirrezaV1/emotional_model", num_labels=7):
    model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

import torch

def predict_emotion_label(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run the model
    outputs = model(**inputs)
    
    # Get the logits and predict the label
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    # Get the final label
    final_label = label_dict[predicted_label]
    prob = torch.nn.functional.softmax(logits, dim=-1).max().item()
    return final_label, prob


def predict_emotion_of_texts(texts: list, model, tokenizer, max_length=128):
    model.eval()
    num_classes = len(label_dict)
    preds = torch.zeros(num_classes).to(device)

    for i, text in enumerate(texts):
        encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds += logits.sum(dim=0)  # Sum across the num_classes dimension

    preds /= len(texts)

    # Apply softmax for probability-like scores
    preds = F.softmax(preds, dim=0)

    # Use the predicted index directly
    _, label_idx = torch.max(preds, dim=0)
    label = label_dict[label_idx.item()]

    return label, preds.tolist()
