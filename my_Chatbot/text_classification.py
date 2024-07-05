import pandas as pd
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, RobertaModel
from sklearn.metrics import classification_report
from tqdm import tqdm
from sklearn.metrics import f1_score

import config


def preprocess_text(sen: str) -> str:
    """
        This function removes punctuations, numbers, single characters, and multiple spaces from a text
        :param sen: a text to preprocess
        :return: a preprocessed text
    """
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def load_data(filename: str) -> pd.DataFrame:
    """
        This function loads the data from a csv file and filters out the empty or null texts
        :param filename: the name of the csv file
        :return: a pandas dataframe with the data
    """
    # Load the data from the csv file
    data = pd.read_csv(filename)

    # Filter out the empty or null texts
    filter = (data["post_text"] != "") & (data["post_text"].notnull())
    data = data[filter]
    data = data.dropna()

    return data


def split_data(data: pd.DataFrame, label_dict: dict, test_size: float) -> tuple:
    """
        This function splits the data into texts and labels, and then into train and test sets
        :param data: a pandas dataframe with the data
        :return: a tuple of four lists: X_train, X_test, y_train, y_test
    """
    # Extract the texts and labels from the data
    X = []
    sentences = list(data["post_text"])
    for sen in sentences:
        X.append(preprocess_text(sen))

    y = data[label_dict.keys()].values

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE)

    return X_train, X_test, y_train, y_test


def to_tensor(X_train, X_test, y_train, y_test):
    """
        This function converts the splited data into tensors
    """
    data = {
        'train': {'texts': X_train, 'labels': torch.as_tensor(y_train)},
        'test': {'texts': X_test, 'labels': torch.as_tensor(y_test)},
    }
    return data


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
    def __init__(self, bert_model_name, num_classes, bert_class):
        super(BERTClassifier, self).__init__()
        self.bert = bert_class.from_pretrained(bert_model_name)
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


def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    predictions = []
    actual_labels = []
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for batch in tqdm(data_loader, position=0):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        preds = torch.sigmoid(outputs) >= 0.5
        predictions.extend(preds.cpu().tolist())
        actual_labels.extend(labels.cpu().tolist())
    return f1_score(actual_labels, predictions, average='samples'), classification_report(actual_labels, predictions)


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, position=0):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            # labels_one_hot = one_hot(labels, num_classes).float()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs) >= 0.5
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return f1_score(actual_labels, predictions, average='samples'), classification_report(actual_labels, predictions)


def predict_label(text, model, tokenizer, device, label_dict, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.sigmoid(outputs) >= 0.5
        preds = preds.type(torch.uint8)
        if 1 not in preds:
            label = 'OTHER'
        else:
            _, label = torch.max(outputs, dim=1)
            label = list(label_dict.keys())[label.item()]
        return label, preds


class TrainingConfig:
    def __init__(self, max_length: int, batch_size: int, learning_rate: float, num_epochs: int) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


def main(data, language_model, training_configs: TrainingConfig, label_dict: dict, layer=None, tokenizer_class=BertTokenizer, bert_class=BertModel):
    tokenizer = tokenizer_class.from_pretrained(language_model)
    train_dataset = TextClassificationDataset(data['train']['texts'], data['train']['labels'], tokenizer, training_configs.max_length)
    val_dataset = TextClassificationDataset(data['test']['texts'], data['test']['labels'], tokenizer, training_configs.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=training_configs.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=training_configs.batch_size)

    num_classes = len(label_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BERTClassifier(language_model, num_classes, bert_class).to(device)

    if layer:
        for name, param in model.named_parameters():
            if layer in name: 
                break
            param.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=training_configs.learning_rate)
    total_steps = len(train_dataloader) * training_configs.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    reports = {'train': list(), 'val': list()}
    for epoch in range(training_configs.num_epochs):
        print(f"Epoch {epoch + 1}/{training_configs.num_epochs}")
        train_accuracy, train_report = train(model, train_dataloader, optimizer, scheduler, device)
        train(model, train_dataloader, optimizer, scheduler, device)
        val_accuracy, val_report = evaluate(model, val_dataloader, device)
        print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print('-'*20)
        reports['val'].append([val_accuracy, val_report])
        reports['train'].append([train_accuracy, train_report])
    print(val_report)
    return reports, model, tokenizer, device


def plot_report(reports):
    val = [i[0] for i in reports['val']]
    train = [i[0] for i in reports['train']]
    x = range(len(val))
    plt.plot(x, train, label='train')
    plt.plot(x, val, label='val')
    plt.legend()
