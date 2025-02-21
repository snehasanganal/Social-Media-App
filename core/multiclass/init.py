import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertConfig

# Define constants
TOKENIZER = "bert-base-cased"
MAX_TOKEN_COUNT = 50
EPOCHS = 2
HIDDEN_SIZE = 256  # Based on Paper 6
NUM_LSTM_LAYERS = 2  # As per Paper 6
DROPOUT_RATE = 0.2
labels = ['Health', 'Politics', 'Religion', 'Sexuality','Location','Personal Information']  # ✅ Multi-Class Labels

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [labels.index(label) for label in df['label']]
        self.texts = [tokenizer(text, padding='max_length', max_length=MAX_TOKEN_COUNT, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], np.array(self.labels[idx])


class PrivacyBERTLSTM_MultiClass(torch.nn.Module):
    def __init__(self, learning_rate, batch_size, optimizer_choice):
        super(PrivacyBERTLSTM_MultiClass, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set the dropout rate for BERT to 0.5
        config = BertConfig.from_pretrained(TOKENIZER)
        config.hidden_dropout_prob = 0.5  # Set dropout probability to 0.5
        self.bert = BertModel.from_pretrained(TOKENIZER).to(self.device)
        self.lstm = torch.nn.LSTM(input_size=768, hidden_size=HIDDEN_SIZE, num_layers=NUM_LSTM_LAYERS,
                                  batch_first=True, bidirectional=True, dropout=DROPOUT_RATE).to(self.device)  # ✅ Fixed Dropout = 0.2
        self.attention = torch.nn.Linear(HIDDEN_SIZE * 2, 1).to(self.device)
        self.fc = torch.nn.Linear(HIDDEN_SIZE * 2, 6).to(self.device)  # Multi-Class Output (4 classes)
        self.activation = torch.nn.Softmax(dim=1).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.batch_size = batch_size

        # Optimizer selection
        if optimizer_choice == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_choice == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)  # Move input tensors to the correct device
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        lstm_output, _ = self.lstm(bert_output.last_hidden_state)

        # Attention mechanism
        attention_weights = torch.nn.functional.softmax(self.attention(lstm_output), dim=1)
        attention_output = torch.sum(attention_weights * lstm_output, dim=1)

        logits = self.fc(attention_output)
        return self.activation(logits)

    def fit(self, train_data):
        self.train()
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        for epoch in range(EPOCHS):
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(self.device)  # Move labels to device
                mask = train_input['attention_mask'].to(self.device)
                input_ids = train_input['input_ids'].squeeze(1).to(self.device)

                output = self(input_ids, mask)
                loss = self.criterion(output, train_label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, val_data, model_name):
        self.eval()
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.batch_size)
        y_pred, y_true = [], []
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(self.device)
                mask = val_input['attention_mask'].to(self.device)
                input_ids = val_input['input_ids'].squeeze(1).to(self.device)

                output = self(input_ids, mask)
                batch_loss = self.criterion(output, val_label)
                total_loss_val += batch_loss.item()

                y_true.extend(val_label.cpu().numpy())
                y_pred.extend(output.argmax(dim=1).cpu().numpy())

        # ✅ Compute Multi-Class Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        loss = total_loss_val / len(val_dataloader)  # ✅ Compute Average Loss

        # ✅ Compute Ranking Score
        ranking_score = (0.4 * f1_weighted) + (0.2 * accuracy) + (0.2 * precision) + (0.2 * recall)

        # ✅ Compute Confusion Matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix, index=labels, columns=labels)
        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, annot=True, fmt='d')
        cm_path = f"/content/{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        print(f"✅ Confusion Matrix saved at {cm_path}")
        
        # ✅ Save Results
        results_path = f"/content/{model_name}_evaluation.txt"
        with open(results_path, "w") as w:
            w.write(f"Accuracy: {accuracy:.3f}\n")
            w.write(f"Loss: {loss:.3f}\n")
            w.write(f"Precision (Weighted): {precision:.3f}\n")
            w.write(f"Recall (Macro): {recall:.3f}\n")
            w.write(f"F1-score (Weighted): {f1_weighted:.3f}\n")
            w.write(f"Final Ranking Score: {ranking_score:.3f}\n")

        print(f"✅ Evaluation results saved at {results_path}")

        return ranking_score, accuracy, precision, recall, f1_weighted, loss  # ✅ Return All Metrics
