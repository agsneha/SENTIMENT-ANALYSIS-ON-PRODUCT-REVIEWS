import streamlit as st
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Model Evaluation")

@st.cache_resource
def load_models_and_vectorizer():
    try:
        base_dir = "/Users/snehaagrawal/Documents/SEM 2/Web Mining"
        models_dict = {
            "logistic_regression": "logistic_regression.pkl",
            "naive_bayes": "naive_bayes.pkl",
            "svm": "support_vector_machine.pkl",
            "vectorizer": "tfidf_vectorizer.pkl",
            "bert_model": "bert_model.pt",
            "roberta_model": "roberta_model.pt"
        }

        # Load non-neural models and vectorizer
        models = {}
        for key, filename in models_dict.items():
            filepath = os.path.join(base_dir, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"{key} file not found at {filepath}")

            if key in ["logistic_regression", "naive_bayes", "svm", "vectorizer"]:
                with open(filepath, 'rb') as file:
                    models[key] = pickle.load(file)
            elif key == "bert_model":
                bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
                bert_model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
                bert_model.eval()
                models[key] = bert_model
            elif key == "roberta_model":
                roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
                roberta_model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
                roberta_model.eval()
                models[key] = roberta_model

        return models["logistic_regression"], models["naive_bayes"], models["svm"], models["vectorizer"], models[
            "bert_model"], models["roberta_model"]
    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading models or vectorizer: {str(e)}")
        return None, None, None, None, None, None


test_data = pd.read_csv('/Users/snehaagrawal/Documents/SEM 2/Web Mining/test_data.csv')
test_data = test_data.dropna(subset=['TRANSLATED_TEXT'])
X_test = test_data['TRANSLATED_TEXT']
y_test = test_data['SENTIMENT']
logistic_regression_model, naive_bayes_model, svm_model, vectorizer, bert_model, roberta_model = load_models_and_vectorizer()

if vectorizer:
    X_test_vectorized = vectorizer.transform(X_test)
    logistic_regression_predictions = logistic_regression_model.predict(X_test_vectorized)
    naive_bayes_predictions = naive_bayes_model.predict(X_test_vectorized)
    svm_predictions = svm_model.predict(X_test_vectorized)

def get_transformer_predictions(model, tokenizer, input_texts):
    max_len = 128
    input_ids = []
    attention_masks = []
    for text in input_texts:
        encoded_dict = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=max_len, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs[0]
        predictions = torch.argmax(logits, dim=1).numpy()
    return predictions


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

models = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "BERT", "RoBERTa"]
results = {}

# Perform cross-validation for each model
for model_name, model in zip(models, [logistic_regression_model, naive_bayes_model, svm_model, bert_model, roberta_model]):
    if model_name in ["Logistic Regression", "Naive Bayes", "Support Vector Machine"]:
        X_test_vectorized = vectorizer.transform(X_test)
        cv_scores = cross_val_score(model, X_test_vectorized, y_test, cv=5, scoring='accuracy')
        model_predictions = model.predict(X_test_vectorized)
    else:
        tokenizer = bert_tokenizer if model_name == "BERT" else roberta_tokenizer
        max_len = 128
        input_ids = []
        attention_masks = []
        for text in X_test:
            encoded_dict = tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=max_len, padding='max_length',
                truncation=True, return_attention_mask=True, return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, torch.tensor(y_test.values))
        batch_size = 16
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        cv_scores = []
        for _ in range(5):  # 5-fold cross-validation
            model.eval()
            fold_predictions = []
            for batch in dataloader:
                batch_input_ids = batch[0].to('cpu')
                batch_attention_masks = batch[1].to('cpu')
                with torch.no_grad():
                    outputs = model(batch_input_ids, attention_mask=batch_attention_masks)
                    logits = outputs[0]
                    fold_predictions.extend(torch.argmax(logits, dim=1).tolist())
            cv_scores.append(accuracy_score(y_test, fold_predictions))
        model_predictions = get_transformer_predictions(model, tokenizer, X_test)

    accuracy = np.mean(cv_scores)
    precision = precision_score(y_test, model_predictions, average=None)
    recall = recall_score(y_test, model_predictions, average=None)
    f1 = f1_score(y_test, model_predictions, average=None)
    cm = confusion_matrix(y_test, model_predictions)
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, model_predictions == i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    results[model_name] = {
        "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1,
        "Confusion Matrix": cm, "FPR": fpr, "TPR": tpr, "ROC AUC": roc_auc
    }

st.title("Model Evaluation")

# Add user input for sentiment prediction
st.subheader("Sentiment Prediction")
user_input = st.text_input("Enter text for sentiment prediction:")

if user_input:
    model_predictions = {}
    for model_name, model in zip(models,
                                 [logistic_regression_model, naive_bayes_model, svm_model, bert_model, roberta_model]):
        if model_name in ["Logistic Regression", "Naive Bayes", "Support Vector Machine"]:
            user_input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(user_input_vectorized)[0]
        else:
            tokenizer = bert_tokenizer if model_name == "BERT" else roberta_tokenizer
            prediction = get_transformer_predictions(model, tokenizer, [user_input])[0]

        model_predictions[model_name] = prediction
        st.write(f"{model_name} Prediction: {prediction}")

    # Compare model predictions
    st.subheader("Model Prediction Comparison")
    comparison_data = {
        "Model": list(model_predictions.keys()),
        "Prediction": list(model_predictions.values())
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

model_options = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "BERT", "RoBERTa", "Model Comparison", "Best Model"]
selected_model = st.selectbox("Select a model:", model_options)
if selected_model in models:
    st.subheader(selected_model)
    model_results = results.get(selected_model, None)


    def ensure_full_coverage(metrics, num_classes=3):
        if len(metrics) < num_classes:
            additional_metrics = [0] * (num_classes - len(metrics))
            metrics.extend(additional_metrics)
        return metrics


    # In your main code block, use this function to prepare your metrics data
    if model_results:
        metrics_data = {
            "Class": ["Negative", "Neutral", "Positive"],
            "Accuracy": [model_results["Accuracy"]] * 3,  # Assuming 'Accuracy' is scalar and repeated for each class
            "Precision": ensure_full_coverage(list(model_results["Precision"]), 3),
            "Recall": ensure_full_coverage(list(model_results["Recall"]), 3),
            "F1-Score": ensure_full_coverage(list(model_results["F1-Score"]), 3)
        }

        metrics_df = pd.DataFrame(metrics_data)

        metrics_df.set_index("Class", inplace=True)
        st.table(metrics_df.style.set_properties(**{'text-align': 'center'}).set_table_styles(
            [{'selector': 'th', 'props': [('background-color', '#f0f0f0')]}]))

        with st.expander("Confusion Matrix"):
            st.markdown("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(model_results["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title(f"{selected_model} Confusion Matrix")
            st.pyplot(fig)

            plt.savefig(f'/Users/snehaagrawal/Documents/SEM 2/Web Mining/{selected_model}_confusion_matrix.png')

        with st.expander("ROC Curve"):
            st.markdown("### ROC Curve")
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(3):
                ax.plot(model_results["FPR"][i], model_results["TPR"][i],
                        label=f"Class {i} (AUC = {model_results['ROC AUC'][i]:.2f})")
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{selected_model} ROC Curve")
            ax.legend(loc="lower right")
            st.pyplot(fig)

            plt.savefig(f'/Users/snehaagrawal/Documents/SEM 2/Web Mining/{selected_model}_roc_curve.png')

elif selected_model == "Model Comparison":
    st.subheader("Model Comparison")
    comparison_data = {
        "Model": models,
        "Accuracy": [results.get(model, {}).get("Accuracy", np.nan) for model in models],
        "Precision (Macro)": [np.mean(results.get(model, {}).get("Precision", [np.nan])) for model in models],
        "Recall (Macro)": [np.mean(results.get(model, {}).get("Recall", [np.nan])) for model in models],
        "F1-Score (Macro)": [np.mean(results.get(model, {}).get("F1-Score", [np.nan])) for model in models]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index("Model", inplace=True)
    st.table(comparison_df)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.2
    ax.bar(x - width, comparison_df["Precision (Macro)"], width, label="Precision (Macro)")
    ax.bar(x, comparison_df["Recall (Macro)"], width, label="Recall (Macro)")
    ax.bar(x + width, comparison_df["F1-Score (Macro)"], width, label="F1-Score (Macro)")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    plt.savefig('/Users/snehaagrawal/Documents/SEM 2/Web Mining/model_comparison.png')

elif selected_model == "Best Model":
    st.subheader("Best Model")
    best_model = max(results, key=lambda x: results.get(x, {}).get("Accuracy", -1))
    model_results = results.get(best_model, None)
    if model_results:
        st.write(f"The best model based on accuracy is: {best_model}")
        st.markdown("### Evaluation Metrics")
        metrics_data = {
            "Class": ["Negative", "Neutral", "Positive"],
            "Accuracy": [model_results["Accuracy"]] * 3,
            "Precision": model_results["Precision"],
            "Recall": model_results["Recall"],
            "F1-Score": model_results["F1-Score"]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.set_index("Class", inplace=True)
        st.table(metrics_df)
else:
    st.warning("Invalid option selected.")

