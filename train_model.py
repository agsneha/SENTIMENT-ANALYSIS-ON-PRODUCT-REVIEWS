import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
import pickle
import torch
import transformers
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, AdamW
import warnings

warnings.filterwarnings('ignore')

directory = "/Users/snehaagrawal/Downloads"
file_path = os.path.join(directory, "resampled_table.csv")
df = pd.read_csv(file_path)

df.columns = df.columns.str.upper()
df['SENTIMENT'] = df['SENTIMENT'].astype(int)

# Handle missing values
df['TRANSLATED_TEXT'] = df['TRANSLATED_TEXT'].fillna('')

# Split the data into features and labels
X = df['TRANSLATED_TEXT']
y = df['SENTIMENT']

# Split the data into train and test sets with 80% in train set and 20% in test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the test data
test_data = pd.DataFrame({'TRANSLATED_TEXT': X_test, 'SENTIMENT': y_test})
test_data.to_csv("/Users/snehaagrawal/Documents/SEM 2/Web Mining/test_data.csv", index=False)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Save the vectorizer
with open("/Users/snehaagrawal/Documents/SEM 2/Web Mining/tfidf_vectorizer.pkl", 'wb') as file:
    pickle.dump(vectorizer, file)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": SVC()
}

# Perform cross-validation and train the models
for model_name, model in models.items():
    cv_scores = cross_validate(model, X_train_vectorized, y_train, cv=5, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'], return_train_score=True)
    print(f"{model_name} - Cross-validation scores:")
    print(f"  Train Accuracy: {cv_scores['train_accuracy'].mean():.2f}")
    print(f"  Validation Accuracy: {cv_scores['test_accuracy'].mean():.2f}")
    print(f"  Validation Precision (macro): {cv_scores['test_precision_macro'].mean():.2f}")
    print(f"  Validation Recall (macro): {cv_scores['test_recall_macro'].mean():.2f}")
    print(f"  Validation F1-score (macro): {cv_scores['test_f1_macro'].mean():.2f}")

    # Train the model on the entire training set
    model.fit(X_train_vectorized, y_train)

    # Save the trained model
    with open(f"/Users/snehaagrawal/Documents/SEM 2/Web Mining/{model_name.lower().replace(' ', '_')}.pkl", 'wb') as file:
        pickle.dump(model, file)

# Fine-tune and save BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

max_len = 128
input_ids = []
attention_masks = []

for text in X_train:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(y_train.values)

dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
batch_size = 16
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 2

for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        batch_input_ids = batch[0].to('cpu')
        batch_input_mask = batch[1].to('cpu')
        batch_labels = batch[2].to('cpu')

        model.zero_grad()
        outputs = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask, labels=batch_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "/Users/snehaagrawal/Documents/SEM 2/Web Mining/bert_model.pt")

# Fine-tune and save RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

max_len = 128
input_ids = []
attention_masks = []

for text in X_train:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(y_train.values)

dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
batch_size = 16
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 2

for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        batch_input_ids = batch[0].to('cpu')
        batch_input_mask = batch[1].to('cpu')
        batch_labels = batch[2].to('cpu')

        model.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_input_mask, labels=batch_labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "/Users/snehaagrawal/Documents/SEM 2/Web Mining/roberta_model.pt")