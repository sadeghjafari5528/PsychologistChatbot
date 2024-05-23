from transformers import XLMRobertaForSequenceClassification, AutoTokenizer

def load_model_and_tokenizer(model_name="AmirrezaV1/emotional_model", num_labels=7):
    model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

import torch

def predict_label(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Run the model
    outputs = model(**inputs)
    
    # Get the logits and predict the label
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    # Define label dictionary
    label_dict = {
        0: 'OTHER',
        1: 'HAPPY',
        2: 'SURPRISE',
        3: 'FEAR',
        4: 'HATE',
        5: 'ANGRY',
        6: 'SAD',
    }
    
    # Get the final label
    final_label = label_dict[predicted_label]
    return final_label

# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Sample text
text = "I'm so excited to be learning about NLP!"

# Predict the label for the sample text
label = predict_label(text, model, tokenizer)

print(f"Predicted Label: {label}")
