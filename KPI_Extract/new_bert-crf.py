import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertTokenizerFast,
    AdamW,
    get_scheduler,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import evaluate
from tika import parser
import spacy
import torch.nn.functional as F
import re
import plotly.express as px
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Global label mappings will be set by the dataset loader
id2label = {}
label2id = {}



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        labels = inputs.pop("labels")  # Extract labels
    
        outputs = model(**inputs)
        logits = outputs.logits  # Extract logits
        
        # Create mask for valid tokens (labels != -100)
        valid_mask = labels != -100
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        
        # Apply mask before one-hot encoding
        valid_labels = labels[valid_mask]
        valid_logits = logits.view(-1, logits.shape[-1])[valid_mask.view(-1)]
        
        # Convert valid labels to one-hot encoding
        labels_one_hot = F.one_hot(valid_labels, num_classes=logits.shape[-1]).float()
        
        loss_all = loss_fn(valid_logits, labels_one_hot)
        loss = loss_all.mean()
        
        return (loss, outputs) if return_outputs else loss

###########################################
# Dataset Creation and Token Alignment
###########################################

class NERDataset:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.dataset = None

    def create_dataset(self) -> DatasetDict:
        """
        Reads the JSON file with token and label data, flattens it,
        converts labels to numeric IDs, and splits the data into train/test sets.
        """
        js = pd.read_json(self.json_path, encoding="utf-8")
        tokens_flat = []
        labels_flat = []
        sentence_ids_flat = []

        # Flatten the lists from each sentence
        for _, row in js.iterrows():
            tokens = row["tokens"]
            labels = row["labels"]
            sentence_id = row["sentence_id"]
            if len(tokens) == len(labels):
                tokens_flat.extend(tokens)
                labels_flat.extend(labels)
                sentence_ids_flat.extend([sentence_id] * len(tokens))
            else:
                print(f"Skipping sentence_id {sentence_id} due to mismatched lengths: {len(tokens)} vs {len(labels)}")
                
        assert len(tokens_flat) == len(labels_flat) == len(sentence_ids_flat), "Mismatch in list lengths!"

        global id2label, label2id
        unique_labels = list(set(labels_flat))
        id2label = {idx: label for idx, label in enumerate(unique_labels)}
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        labels_numeric_flat = [label2id[label] for label in labels_flat]

        dataset_dict = {
            "tokens": tokens_flat,
            "ner_tags": labels_flat,
            "sentence_id": sentence_ids_flat,
            "labels_numeric": labels_numeric_flat
        }
        dataset = Dataset.from_dict(dataset_dict)
        dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
        self.dataset = DatasetDict({
            "train": dataset_split["train"],
            "test": dataset_split["test"]
        })

        print("Sample training data:", self.dataset["train"][0])
        print(f"Training set size: {len(self.dataset['train'])}")
        print(f"Test set size: {len(self.dataset['test'])}")
        return self.dataset


class TokenizerAligner:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(self, dataset: DatasetDict) -> DatasetDict:
        def tokenize_fn(batch):
            def align_target(labels, word_ids):
                align_labels = []
                for word in word_ids:
                    if word is None:
                        label = -100
                    else:
                        label = labels[word]
                    align_labels.append(label)
                return align_labels

            tokenized_inputs = self.tokenizer(
                batch["tokens"],
                truncation=True,
                padding=True,
                is_split_into_words=True,
                max_length=512,
                return_tensors="np"
            )
            labels_batch = batch["labels_numeric"]
            aligned_targets_batch = [align_target(labels_batch, tokenized_inputs.word_ids())]
            tokenized_inputs["labels"] = aligned_targets_batch
            return tokenized_inputs

        tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)
        tokenized_datasets.set_format("torch")
        print("Tokenization and alignment completed.")
        return tokenized_datasets

###########################################
# Metrics and Loss Functions
###########################################

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    true_predictions = []
    true_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                true_predictions.append(p)
                true_labels.append(l)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(true_labels, true_predictions)
    # Trainer will prefix metric keys with "eval_"
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs["logits"]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return (loss, outputs) if return_outputs else loss

###########################################
# Trainer Class and Saving Methods
###########################################

class NERTrainer:
    def __init__(self, model_name: str, tokenizer, tokenized_datasets: DatasetDict):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.tokenized_datasets = tokenized_datasets
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)

    def train(self):
        training_args = TrainingArguments(
            output_dir="fine_tuned_model",
            evaluation_strategy="epoch",
            remove_unused_columns=False,
            learning_rate=0.0001,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=300,
            weight_decay=0.01,
            no_cuda=False,
            optim= 'adamw_torch'
        )

        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['test'],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator,
        )

        trainer.train()
        trainer.save_model("fine_tuned_model")

        eval_results = trainer.evaluate()
        print("Evaluation Metrics:")
        print("Precision:", eval_results.get("eval_precision"))
        print("F1 Score:", eval_results.get("eval_f1"))
        print("Accuracy:", eval_results.get("eval_accuracy"))

        # Save metrics to CSV (using model name as file name)
        metrics = {
            "Model": self.model_name,
            "Precision": eval_results.get("eval_precision"),
            "F1 Score": eval_results.get("eval_f1"),
            "Accuracy": eval_results.get("eval_accuracy")
        }
        metrics_df = pd.DataFrame([metrics])
        filename = f"{self.model_name.replace('/', '_')}_metrics.csv"
        metrics_df.to_csv(filename, index=False)
        print(f"Metrics saved to {filename}")

        return trainer

    def save_and_push_model(self, local_dir: str, hub_repo: str = None, commit_message: str = "Initial commit"):
        # Save locally
        self.model.save_pretrained(local_dir)
        self.tokenizer.save_pretrained(local_dir)
        print(f"Model and tokenizer saved locally at {local_dir}")
        # Push to Hugging Face Hub if hub_repo is provided
        if hub_repo is not None:
            self.model.push_to_hub(hub_repo, commit_message=commit_message)
            self.tokenizer.push_to_hub(hub_repo, commit_message=commit_message)
            print(f"Model and tokenizer pushed to Hugging Face Hub repository: {hub_repo}")

###########################################
# Main Function
###########################################

def main():
    model_name = "ProsusAI/finbert"
    json_path = "Augmented_Annotated_JSON_1103.json"  # Path to your JSON file

    # Create the dataset
    dataset_loader = NERDataset(json_path)
    dataset = dataset_loader.create_dataset()

    # Tokenize and align labels
    tokenizer_aligner = TokenizerAligner(model_name)
    tokenized_datasets = tokenizer_aligner.tokenize_and_align_labels(dataset)

    # Optionally inspect a sample
    sample_tokens = tokenized_datasets['train']['input_ids'][0]
    print("Sample tokenized input_ids shape:", sample_tokens.shape if hasattr(sample_tokens, "shape") else "N/A")

    # Train the model
    ner_trainer = NERTrainer(model_name, tokenizer_aligner.tokenizer, tokenized_datasets)
    trainer = ner_trainer.train()

    # Save model locally and push to HF Hub (if desired)
    ner_trainer.save_and_push_model(local_dir="bert_crf", hub_repo="Nidhilakhan-17/nl_kpi_ner_bert_crf", commit_message="CRF-BERT")

if __name__ == "__main__":
    main()
