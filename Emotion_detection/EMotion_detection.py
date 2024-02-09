! git clone https://github.com/Arman-Rayan-Sharif/arman-text-emotion.git
DATA_PATH = '/content/arman-text-emotion/dataset'
!pip install transformers
from transformers import XLMRobertaModel
!pip install --upgrade pip
# !pip uninstall openai
!pip install openai
!pip install sentencepiece
!pip install ntscraper
!pip install typing_extensions --upgrade
!pip install transformers==4.11.3
!pip install sentencepiece
from google.colab import drive
drive.mount('/content/drive')
# from ntscraper import Nitter
from ntscraper import Nitter
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from torch import nn

import torch
import torch.nn.functional as F
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, RobertaModel, RobertaTokenizer, XLMRobertaTokenizer
from transformers import XLMRobertaModel  # Add this line to import XLMRobertaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
import pandas as pd

label_dict = {
  'OTHER': 0,
  'HAPPY': 1,
  'SURPRISE': 2,
  'FEAR': 3,
  'HATE': 4,
  'ANGRY': 5,
  'SAD': 6,
}
train_df = pd.read_table(f'{DATA_PATH}/train.tsv', header=None)
train_df[1] = train_df[1].map(label_dict)
train_texts, train_labels = train_df[0], train_df[1]

test_df = pd.read_table(f'{DATA_PATH}/test.tsv', header=None)
test_df[1] = test_df[1].map(label_dict)
test_texts, test_labels = test_df[0], test_df[1]
data = {
    'train': {'texts': train_texts, 'labels': train_labels},
    'test': {'texts': test_texts, 'labels': test_labels},
}


def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Apply the clean_text function to your texts
data['train']['texts'] = data['train']['texts'].apply(clean_text)
data['test']['texts'] = data['test']['texts'].apply(clean_text)

class TextClassificationDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_length):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    text = self.texts[idx]
    label = self.labels[idx]
    encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
    return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
class BERTClassifier(nn.Module):
  def __init__(self, bert_model_name, num_classes):
    super(BERTClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(bert_model_name)
    self.dropout = nn.Dropout(0.1)
    self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.pooler_output
    x = self.dropout(pooled_output)
    logits = self.fc(x)
    return logits
class RobertaClassifier(nn.Module):
  def __init__(self, bert_model_name, num_classes):
    super(RobertaClassifier, self).__init__()
    self.bert = RobertaModel.from_pretrained(bert_model_name)
    self.dropout = nn.Dropout(0.1)
    self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.pooler_output
    x = self.dropout(pooled_output)
    logits = self.fc(x)
    return logits

class XLMRobertaClassifier(nn.Module):
  def __init__(self, model_name, num_classes):
    super(XLMRobertaClassifier, self).__init__()
    self.bert = XLMRobertaModel.from_pretrained(model_name)
    self.dropout = nn.Dropout(0.1)
    self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = outputs.pooler_output
    x = self.dropout(pooled_output)
    logits = self.fc(x)
    return logits

def train(model, data_loader, optimizer, scheduler, device):
  model.train()
  predictions = []
  actual_labels = []
  for batch in tqdm(data_loader, position=0):
    optimizer.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    _, preds = torch.max(outputs, dim=1)
    predictions.extend(preds.cpu().tolist())
    actual_labels.extend(labels.cpu().tolist())
  return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def predict_emoji(texts, model, tokenizer, device, label_dict, max_length=128):
    model.eval()
    num_classes = len(label_dict)
    preds = torch.zeros(num_classes).to(device)
    linear = torch.nn.Linear(model.fc.in_features, num_classes).to(device)

    for i, text in enumerate(texts):
        encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds += torch.sigmoid(logits).sum(dim=0)  # Sum across the num_classes dimension

    preds /= len(texts)

    # Apply softmax for probability-like scores
    preds = F.softmax(preds, dim=0)

    # Use the predicted index directly
    _, label_idx = torch.max(preds, dim=0)
    label = label_dict[label_idx.item()]

    return label, preds.tolist()
def evaluate(model, data_loader, device):
  model.eval()
  predictions = []
  actual_labels = []
  with torch.no_grad():
    for batch in tqdm(data_loader, position=0):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      _, preds = torch.max(outputs, dim=1)
      predictions.extend(preds.cpu().tolist())
      actual_labels.extend(labels.cpu().tolist())
  return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
num_classes = 7
max_length = 128
batch_size = 16
num_epochs = 5
learning_rate = 2e-5

def main(data, language_model, layer=None, tokenizer_class=BertTokenizer, classifier_class=XLMRobertaClassifier):
    tokenizer = tokenizer_class.from_pretrained(language_model)
    train_dataset = TextClassificationDataset(data['train']['texts'], data['train']['labels'], tokenizer, max_length)
    val_dataset = TextClassificationDataset(data['test']['texts'], data['test']['labels'], tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the classifier class
    model = classifier_class(language_model, num_classes).to(device)

    if layer:
        # freeze bert parameters
        for name, param in model.named_parameters():
            if layer in name:  # classifier layer
                break
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    reports = {'train': list(), 'val': list()}
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_accuracy, train_report = train(model, train_dataloader, optimizer, scheduler, device)
        val_accuracy, val_report = evaluate(model, val_dataloader, device)
        print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print('-' * 20)
        reports['val'].append([val_accuracy, val_report])
        reports['train'].append([train_accuracy, train_report])
    print(val_report)
    return model, reports

xlm_roberta_model = 'xlm-roberta-large'
xlm_roberta_model_instance, xlm_roberta_reports = main(data, xlm_roberta_model, layer='11', tokenizer_class=XLMRobertaTokenizer, classifier_class=XLMRobertaClassifier)
#

torch.save(xlm_roberta_model_instance, "/content/drive/My Drive/emotional-main-model-.pth")

def get_tweets(name, modes, no):
    scraper = Nitter(0)
    tweets = scraper.get_tweets(name, mode=modes, number=no)
    final_tweets = []
    for x in tweets['tweets']:
        data = [name, x['link'], x['text'], x['date'], x['stats']['likes'], x['stats']['comments']]
        final_tweets.append(data)
    dat = pd.DataFrame(final_tweets, columns=['user_id', 'twitter_link', 'text', 'date', 'likes', 'comments'])
    return dat

# List of IDs
id_list = ["SharifiZarchi", "Kimiya_Hosseini"]

# Emotion prediction
label_dict = {
    0: 'OTHER',
    1: 'HAPPY',
    2: 'SURPRISE',
    3: 'FEAR',
    4: 'HATE',
    5: 'ANGRY',
    6: 'SAD',
}

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Predict emotions for each user's tweets as a whole
for id in id_list:
    dat = get_tweets(id, 'hashtag', 10)
    user_tweets = dat['text'].tolist()
    label, predictions = predict_emoji(user_tweets, xlm_roberta_model_instance, tokenizer, device, label_dict)
    print(f"User ID: {id}")
    print("Predicted Label:", label)
    print("Predictions:", predictions)
    print("\n")
