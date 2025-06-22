import json
import pandas as pd
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
# Uncomment if you need CRF functionality
# from TorchCRF import CRF

# Global label mappings will be set by the dataset loader
id2label = {}
label2id = {}

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
            sentence_id = row["sentence_id"]  # This is an integer
            if len(tokens) == len(labels):
                tokens_flat.extend(tokens)
                labels_flat.extend(labels)
                sentence_ids_flat.extend([sentence_id] * len(tokens))
            else:
                print(f"Skipping sentence_id {sentence_id} due to mismatched lengths: {len(tokens)} tokens vs {len(labels)} labels")

        # Ensure list lengths match
        assert len(tokens_flat) == len(labels_flat) == len(sentence_ids_flat), "Mismatch in list lengths!"

        # Create global label mappings
        global id2label, label2id
        unique_labels = list(set(labels_flat))
        id2label = {idx: label for idx, label in enumerate(unique_labels)}
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        labels_numeric_flat = [label2id[label] for label in labels_flat]

        # Create a dictionary and then a Hugging Face Dataset
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

        # Display a sample and dataset sizes
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
                # This helper function aligns original labels with tokenized word IDs.
                align_labels = []
                last_word = None
                for word in word_ids:
                    if word is None:
                        label = -100  # For special tokens like [CLS] and [SEP]
                    else:
                        label = labels[word]
                    align_labels.append(label)
                    last_word = word
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
            # For batched input, process each instance separately
            aligned_targets_batch = [align_target(labels_batch, tokenized_inputs.word_ids())]
            tokenized_inputs["labels"] = aligned_targets_batch
            return tokenized_inputs

        tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=dataset['train'].column_names)
        tokenized_datasets.set_format("torch")
        print("Tokenization and alignment completed.")
        return tokenized_datasets


def compute_loss(model, inputs, return_outputs=False):
    """
    Compute cross-entropy loss for a model's token classification output.
    """
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs["logits"]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
    return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    # Placeholder for metrics calculation (e.g., using seqeval for NER)
    # eval_pred is a tuple (predictions, labels)
    # Return a dictionary with metric names and values.
    return {}


class NERTrainer:
    def __init__(self, model_name: str, tokenizer, tokenized_datasets: DatasetDict):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.tokenized_datasets = tokenized_datasets
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.device = "cpu"  # Change to "cuda" if GPU is available

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
            learning_rate=0.001,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            no_cuda=True  # Set to False if using a GPU
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets['train'],
            eval_dataset=self.tokenized_datasets['test'],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator,
        )

        trainer.train()
        return trainer


def main():
    # Define model and data paths
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    json_path = "Annoted_JSON_090225.json"  # Path to your JSON file

    # Create dataset
    dataset_loader = NERDataset(json_path)
    dataset = dataset_loader.create_dataset()

    # Tokenize and align labels
    tokenizer_aligner = TokenizerAligner(model_name)
    tokenized_datasets = tokenizer_aligner.tokenize_and_align_labels(dataset)

    # Optionally, inspect one tokenized example
    sample_tokens = tokenized_datasets['train']['input_ids'][0]
    print("Sample tokenized input_ids shape:", sample_tokens.shape if hasattr(sample_tokens, "shape") else "N/A")

    # Check label range
    labels_tensor = tokenized_datasets["train"]['labels']
    if any(label < -100 or label > max(label2id.values()) for label in labels_tensor.flatten().tolist()):
        print("❌ Label values are out of range!")

    # Train the model
    ner_trainer = NERTrainer(model_name, tokenizer_aligner.tokenizer, tokenized_datasets)
    trainer = ner_trainer.train()


# if __name__ == "__main__":
main()
 