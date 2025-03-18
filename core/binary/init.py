import os
import re
import numpy as np
import pandas as pd
import torch
import spacy
import time
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, AdamW
import nltk
from nltk.corpus import stopwords
import requests
# Load Named Entity Recognition (NER) model
nltk.download('stopwords')

# Load SpaCy's Named Entity Recognition (NER) model
nlp = spacy.load("en_core_web_sm")

# ✅ Load NLTK Stopwords (Common function words)
STOPWORDS = set(stopwords.words('english'))

# Define constants
TOKENIZER = "bert-base-cased"
MAX_TOKEN_COUNT = 50
EPOCHS = 2
HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
DROPOUT_RATE = 0.2
labels = ['Other', 'Sensitive']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)

# Define non-sensitive words and common locations to avoid false positives
NON_SENSITIVE_WORDS = {
    "day", "happy", "joy", "hello", "morning", "night", "fun", "great", "amazing", "exciting",
    "fantastic", "wonderful", "awesome", "beautiful", "cool", "nice", "sunny", "cloudy", "rainy", "bright",
    "relaxing", "energetic", "smile", "laugh", "playing", "singing", "music", "dance", "painting",
    "reading", "learning", "watching","enjoying", "thinking", "cheering",
    "sad", "love", "friend","joy", "fun", "good", "bad", "beautiful",
    "amazing", "crazy", "excited", "boring", "rainy", "sunny", "cloudy",
    "great", "awful", "favorite", "you", "me", "we", "us", "them", "they", "he", "she", "it","her","him","his"
}
def get_world_locations():
    """Fetch a large list of world cities/countries from GeoNames API."""
    try:
        response = requests.get("https://download.geonames.org/export/dump/cities15000.zip")  # GeoNames dataset
        locations = set()
        for line in response.text.split("\n"):
            parts = line.split("\t")
            if len(parts) > 1:
                locations.add(parts[1].lower())  # Extract city names
        
    except Exception as e:
        print(f"⚠️ Failed to fetch world locations: {e}")
    return locations

COMMON_LOCATIONS = get_world_locations()
SENSITIVE_NUMBER_KEYWORDS = {
    "phone", "phn", "mobile", "cell", "contact", "ssn", "social security", "credit card", "debit card",
    "cvv", "pin", "atm", "bank", "account", "acc", "passport", "id number", "driver's license",
    "insurance number", "aadhaar", "upi", "transaction", "routing number", "swift code", "tax id", "iban"
}

# ✅ Regex patterns for detecting sensitive numbers and age
PHONE_REGEX = r"\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"  # Phone numbers
SSN_REGEX = r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"  # Social Security Numbers (SSN)
CREDIT_CARD_REGEX = r"\b(?:\d[ -]*?){13,16}\b"  # Credit/debit card numbers
ACCOUNT_NUMBER_REGEX = r"\b(?:\d[ -]*?){6,12}\b"  # Bank account numbers
GENERAL_NUMBER_REGEX = r"\b\d{4,}\b"  # Generic numbers (ignored unless context is found)
PAN_NUMBER_REGEX = r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
PASSPORT_NUMBER_REGEX = r"\b[A-Z]{1}[0-9]{7}\b"

# ✅ Define question words (to prevent misclassification of general questions)
QUESTION_WORDS = {"what", "where", "why", "when", "how", "who", "which"}

def is_sensitive_number(text):
    """
    Checks if a number is truly sensitive by ensuring it appears with a keyword.
    """
    text = str(text)
    words = text.lower().split()
    if any(keyword in words for keyword in SENSITIVE_NUMBER_KEYWORDS):
        if re.search(PHONE_REGEX, text) or re.search(SSN_REGEX, text) or re.search(CREDIT_CARD_REGEX, text) or re.search(ACCOUNT_NUMBER_REGEX, text) or re.search(PAN_NUMBER_REGEX, text) or re.search(PASSPORT_NUMBER_REGEX, text):
            return 1  # Sensitive if number appears with the correct keyword
    
    return 0

def extract_entities(text):
    """
    Extract named entities, but avoid false positives by checking context.
    """
    doc = nlp(text.lower())
    entities = {ent.label_: ent.text for ent in doc.ents}
    words = set(text.lower().split())

    if len(words) == 1 and (words & STOPWORDS or words & NON_SENSITIVE_WORDS or words & COMMON_LOCATIONS):
        return 0

    if words & QUESTION_WORDS and not entities:
        return 0

    for ent_label, ent_text in entities.items():
        if ent_label in ["DATE"] or re.search(r"\b\d{1,3}\s?(?:years old|yrs old|born in|born on|age)\b", text):
            return 1
        if ent_label in ["PERSON"]:
            return 1
        if ent_label == "GPE" and len(words) > 1:
            return 1

    if is_sensitive_number(text):
        return 1

    return 0

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels.index(label) for label in df['label']]
        self.texts = [tokenizer(text, padding='max_length', max_length=MAX_TOKEN_COUNT, truncation=True,
                                return_tensors="pt") for text in df['text']]
        self.entity_flags = [extract_entities(text) for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], np.array(self.labels[idx]), np.array(self.entity_flags[idx])

class PrivacyBERTLSTM(torch.nn.Module):
    def __init__(self, learning_rate, batch_size, optimizer_choice):
        super(PrivacyBERTLSTM, self).__init__()

        self.device = device
        self.bert = BertModel.from_pretrained(TOKENIZER).to(self.device)
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=HIDDEN_SIZE, num_layers=NUM_LSTM_LAYERS,
                                  batch_first=True, bidirectional=True, dropout=DROPOUT_RATE).to(self.device)
        self.fc = torch.nn.Linear(HIDDEN_SIZE * 2 + 1, 2).to(self.device)
        self.activation = torch.nn.Sigmoid().to(self.device)
        class_weights = torch.tensor([0.4, 0.6]).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.batch_size = batch_size
        
        if optimizer_choice == 'AdamW':
            self.optimizer = AdamW(self.parameters(), lr=learning_rate)
        elif optimizer_choice == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_choice == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, input_ids, attention_mask, entity_flags):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        entity_flags = entity_flags.to(self.device).float().unsqueeze(1)

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        lstm_output, _ = self.lstm(cls_embedding.unsqueeze(1))

        lstm_output = torch.cat((lstm_output[:, -1, :], entity_flags), dim=1)
        logits = self.fc(lstm_output)
        return self.activation(logits)
    """def fit(self, train_data):
        self.train()
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        print("Training started...")
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            
            for train_input, train_label, entity_flags in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
                train_label = train_label.to(self.device)
                mask = train_input['attention_mask'].to(self.device)
                input_ids = train_input['input_ids'].squeeze(1).to(self.device)
                
                output = self(input_ids, mask, entity_flags)
                loss = self.criterion(output, train_label)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            epoch_end = time.time()
            print(f"Epoch {epoch+1} completed in {epoch_end - epoch_start:.2f} seconds\n")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds.")
        torch.save(self.state_dict(), "/content/best_model.pt")"""


    def fit(self, train_data):
        """Trains the model and plots the loss per epoch."""
        self.train()
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        print("Training started...")

        start_time = time.time()
        loss_per_epoch = []  # ✅ List to store loss for each epoch

        for epoch in range(EPOCHS):
            epoch_start = time.time()
            print(f"Epoch {epoch+1}/{EPOCHS}:")

            total_loss = 0  # ✅ Track total loss for this epoch

            for train_input, train_label, entity_flags in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
                train_label = train_label.to(self.device)
                mask = train_input['attention_mask'].to(self.device)
                input_ids = train_input['input_ids'].squeeze(1).to(self.device)
                entity_flags = entity_flags.to(self.device).float()

                output = self(input_ids, mask, entity_flags)
                loss = self.criterion(output, train_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()  # ✅ Accumulate loss for this epoch

            avg_loss = total_loss / len(train_dataloader)
            loss_per_epoch.append(avg_loss)  # ✅ Store average loss per epoch

            epoch_end = time.time()
            print(f"Epoch {epoch+1} completed in {epoch_end - epoch_start:.2f} seconds | Loss: {avg_loss:.4f}\n")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds.")
        torch.save(self.state_dict(), "/content/best_model.pt")

        # ✅ Plot Loss vs. Epochs
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, EPOCHS + 1), loss_per_epoch, marker='o', linestyle='-', color='b', label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss vs. Number of Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()


    def evaluate(self, val_data, model_name):
        self.eval()
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)
        y_pred, y_true = [], []
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label, entity_flags in val_dataloader:
                val_label = val_label.to(self.device)
                mask = val_input['attention_mask'].to(self.device)
                input_ids = val_input['input_ids'].squeeze(1).to(self.device)
                entity_flags = entity_flags.to(self.device).float()

                output = self(input_ids, mask, entity_flags)
                batch_loss = self.criterion(output, val_label)
                total_loss_val += batch_loss.item()

                y_true.extend(val_label.cpu().numpy())
                y_pred.extend(output.argmax(dim=1).cpu().numpy())

        # Compute Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        loss = total_loss_val / len(val_dataloader)

        # Compute Final Ranking Score
        ranking_score = (0.4 * f1_weighted) + (0.2 * accuracy) + (0.2 * precision) + (0.2 * recall)

        # Compute Confusion Matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, index=labels, columns=labels)
        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, annot=True, fmt='d')
        cm_path = f"/content/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        print(f"✅ Confusion Matrix saved at {cm_path}")
        
        # Save Results
        results_path = f"/content/{model_name}_evaluation.txt"
        with open(results_path, "w") as w:
            w.write(f"Accuracy: {accuracy:.3f}\n")
            w.write(f"Loss: {loss:.3f}\n")
            w.write(f"Precision (Weighted): {precision:.3f}\n")
            w.write(f"Recall (Macro): {recall:.3f}\n")
            w.write(f"F1-score (Weighted): {f1_weighted:.3f}\n")
            w.write(f"Final Ranking Score: {ranking_score:.3f}\n")

        print(f"✅ Evaluation results saved at {results_path}")

        return ranking_score, accuracy, precision, recall, f1_weighted, loss  # ✅ Return all key metrics
