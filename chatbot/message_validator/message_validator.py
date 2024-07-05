from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

def load_validator_model_and_tokenizer(model_name="erfanzare/sntiment_txt_classify"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def predict_validator_labels(text, model, tokenizer, threshold=0.5):
    
    inputs = tokenizer(text, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.sigmoid(logits)    
    predicted_labels = (probs > threshold).int().squeeze().tolist()    
    label_dict = {
        0: 'toxic',
        1: 'severe_toxic',
        2: 'obscene',
        3: 'threat',
        4: 'insult',
        5: 'identity_hate',
    }
    
    final_labels = [label_dict[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]    
    return final_labels
