import os
import pandas as pd
import numpy as np
import fitz
import camelot
import nltk
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    pipeline
)
import evaluate
from tika import parser
import spacy
import torch
import torch.nn as nn
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')  # For sent_tokenize and word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')

# Download required NLTK data
nltk.download('punkt')
import torch.nn.functional as F
import re
import plotly.express as px


#     # Custom Trainer class to override compute_loss
# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         loss_fn = nn.CrossEntropyLoss()
        
#         labels = inputs.pop("labels")  # Extract labels
#         outputs = model(**inputs)
#         logits = outputs.logits  # Extract logits from model output

#         loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))  # Compute CrossEntropyLoss

#         return (loss, outputs) if return_outputs else loss


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

class NERPipeline:
    def __init__(self, model_name, json_path, output_dir, num_train_epochs, learning_rate, per_device_train_batch_size, per_device_eval_batch_size, weight_decay, optim, gradient_accumulation_steps):
        """
        Initializes the pipeline with the model checkpoint, path to the JSON data,
        spaCy model name, and output directory for the trained model.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.json_path = json_path
        # Override passed hyperparameters with hardcoded defaults if desired:
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay

        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Placeholders for dataset, label mappings, and model
        self.dataset_final = None
        self.label2id = None
        self.id2label = None
        self.model = None

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.optim = optim

    # ---------------- NERDatasetCreator Functions ----------------
    def create_dataset(self):
        """
        Reads the JSON file with token and label data, flattens it,
        converts labels to numeric IDs, and splits the data into train/test sets.
        
        Returns:
            DatasetDict: A Hugging Face DatasetDict with 'train' and 'test' splits.
        """
        js = pd.read_json(self.json_path, encoding="utf-8")
        tokens_flat = []
        labels_flat = []
        sentence_ids_flat = []
        
        # Flatten the lists
        for _, row in js.iterrows():
            tokens = row["tokens"]
            labels = row["labels"]
            sentence_id = row["sentence_id"]  # This is an integer
        
            # Ensure token-label length matches before extending
            if len(tokens) == len(labels):
                tokens_flat.extend(tokens)  
                labels_flat.extend(labels)  
                sentence_ids_flat.extend([sentence_id] * len(tokens))  # Repeat sentence_id for each token
            else:
                print(f"Skipping sentence_id {sentence_id} due to mismatched lengths: {len(tokens)} tokens vs {len(labels)} labels")
        
        # Ensure final lengths match
        assert len(tokens_flat) == len(labels_flat) == len(sentence_ids_flat), "Mismatch in list lengths!"
        
        # Convert labels to numeric IDs
        unique_labels = list(set(labels_flat))
        self.id2label = {idx: label for idx, label in enumerate(unique_labels)}     
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
        labels_numeric_flat = [self.label2id[label] for label in labels_flat]
        
        # Create Dataset dictionary
        dataset_dict = {
            "tokens": tokens_flat,
            "ner_tags": labels_flat,
            "sentence_id": sentence_ids_flat,
            "labels_numeric": labels_numeric_flat
        }
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Split into train and test sets
        dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Store in DatasetDict format
        self.dataset_final = DatasetDict({
            "train": dataset_split["train"],
            "test": dataset_split["test"]
        })
        
        # Display a sample
        print(self.dataset_final["train"][0])
        print(f"Training set size: {len(self.dataset_final['train'])}")
        print(f"Test set size: {len(self.dataset_final['test'])}")
    
        return self.dataset_final

    # ---------------- TokenizerAligner Functions ----------------
    def tokenize_fn(self, batch):
        """
        Tokenizes the batch of tokens and aligns their labels.
        
        Args:
            batch (dict): Batch with keys "tokens" and "labels_numeric".
            
        Returns:
            dict: Tokenized inputs with aligned labels.
        """
    
        def align_target(labels, word_ids):
            """
            Aligns the labels with the tokenized word IDs.
            
            Args:
                labels (list): Original labels for a sentence.
                word_ids (list): Word IDs from tokenization.
            
            Returns:
                list: Aligned labels (with -100 for special tokens).
            """
            align_labels = []
            last_word = None
            for word in word_ids:
                if word is None:
                    label = -100  
                elif word != last_word:
                    label = labels[word]
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
    
        # Align labels
        labels_batch = batch["labels_numeric"]
    
        print(f"Labels Numeric: {labels_batch}")
    
        aligned_targets_batch = []
        aligned_targets_batch.append(align_target(labels_batch, tokenized_inputs.word_ids()))
    
        # Add the aligned labels to the tokenized inputs
        tokenized_inputs["labels"] = aligned_targets_batch
        print(f"Tokenized Labels: {tokenized_inputs}")
        
        return tokenized_inputs
    
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)  # Get the predicted labels
        accuracy = (predictions == labels).mean()  # Basic accuracy
        return {"eval_accuracy": accuracy}

    def build_model(self):
        """
        Initializes the token classification model with the specified label mappings.
        """
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )
        # You can switch to GPU if available:
        device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)



    def train(self):
        """
        Trains the model using Hugging Face's Trainer.
        """
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="epoch",
            remove_unused_columns=False,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=self.weight_decay,
            no_cuda=False,
            optim=self.optim,
            gradient_accumulation_steps=self.gradient_accumulation_steps
        )
        trainer = CustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=data_collator,
        )
        trainer.train()
        trainer.save_model(self.output_dir)

    def push_model_to_hub(self, repo_name, commit_message="Initial commit"):
        """
        Pushes the trained model and tokenizer to the Hugging Face Hub.
        
        Args:
            repo_name (str): The name of the repository to create/update on the Hugging Face Hub.
            commit_message (str): Commit message for the push.
        """
        try:
            self.model.push_to_hub(repo_name, commit_message=commit_message)
            self.tokenizer.push_to_hub(repo_name, commit_message=commit_message)
            print(f"Model and tokenizer successfully pushed to Hugging Face Hub at '{repo_name}'.")
        except Exception as e:
            print(f"An error occurred while pushing to the hub: {e}")

    def run(self, hub_repo=None):
        """
        Executes the entire pipeline: creates the dataset, tokenizes and aligns labels,
        builds the model, trains it, and optionally pushes it to the Hugging Face Hub.
        
        Args:
            hub_repo (str, optional): If provided, the model will be pushed to this Hugging Face Hub repository.
        """
        self.create_dataset()
        self.tokenized_datasets = self.dataset_final.map(
            self.tokenize_fn,
            batched=True,
            remove_columns=self.dataset_final['train'].column_names
        )
        self.build_model()
        self.train()
        
        # Push to Hugging Face Hub if a repository name is provided

        if hub_repo is not None:
            self.push_model_to_hub(hub_repo)





# --- Post-processing module ---



# --- Updated Inferencing class integrating post-processing ---

class Inferencing:
    def __init__(self, pdf_path, inf_model_name):
        self.pdf_path = pdf_path
        self.inf_model_name = inf_model_name
        
    def extract_tables(self):
        """
        Extracts tables from a PDF file using Camelot.
        """
        pages = "all"
        flavor = "stream"
        tables = camelot.read_pdf(self.pdf_path, pages=pages, flavor=flavor)
        dataframes = [table.df for table in tables]
        return dataframes

    def text_preprocessing(self, max_length=1000):
        """
        Extracts and processes text from a PDF file.
        It also removes stop words in English and German.
        """
        
        report = parser.from_file(self.pdf_path)
        nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])
        nlp.add_pipe("sentencizer")
        text = report["content"]
        # Create chunks to avoid overloading spaCy
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        all_sequences = []
        for chunk in chunks:
            doc = nlp(chunk)
            sequences = list(map(str, doc.sents))
            all_sequences.extend(sequences)
        
        # Initial extraction: filter sentences that are not empty and start with an uppercase letter
        self.sentences = [x.replace("\n", "") for x in all_sequences if x and x[0].isupper()]
        
        # --- Remove stop words in English and German ---
        from spacy.lang.en.stop_words import STOP_WORDS as en_stop
        from spacy.lang.de.stop_words import STOP_WORDS as de_stop
        combined_stop_words = en_stop.union(de_stop)
        
        processed_sentences = []
        for sentence in self.sentences:
            # Split the sentence into words and filter out stop words
            words = sentence.split()
            filtered_sentence = " ".join([word for word in words if word.lower() not in combined_stop_words])
            processed_sentences.append(filtered_sentence)
        
        # Update sentences with the processed ones
        self.sentences = processed_sentences
    
        # Create a dictionary with 'index' as keys and 'value' as values, then convert it to a DataFrame
        sent_dict = {index: value for index, value in enumerate(self.sentences)}
        df = pd.DataFrame(list(sent_dict.items()), columns=["Index", "Sentence"])
        return df

    
    def infer(self, sentence):
        """
        Performs inference using a token-classification pipeline.
        
        Args:
            sentence (str): Input sentence for NER.
            
        Returns:
            list: NER predictions.
        """
        
        inf_tokenizer = AutoTokenizer.from_pretrained(self.inf_model_name)
        inf_model = AutoModelForTokenClassification.from_pretrained(self.inf_model_name)
        
        self.ner_pln = pipeline(
            'token-classification',
            model= inf_model,
            tokenizer= inf_tokenizer
            aggregation_strategy=None
        )
        # return self.ner_pln(sentence)

    
    def extract_word_entity(self, sentence):
        """
        Extracts token predictions from the sentence using the ner_pipeline.
        It combines subword tokens (those starting with "##") with the previous token,
        assigning the KPI of the first token to the entire combined token.
        Returns a comma-separated string of word-entity pairs.
        """
        predictions = self.ner_pln(sentence)
        combined_tokens = []
        
        for pred in predictions:
            token = pred['word']
            entity = pred['entity']
            # If token starts with "##", append it to the previous token (if exists)
            if token.startswith("##") and combined_tokens:
                combined_tokens[-1]['word'] += token[2:]
            else:
                combined_tokens.append({'word': token, 'entity': entity})
        
        # Create comma-separated string of the format "word-entity"
        pairs = [f"{item['word']}-{item['entity']}" for item in combined_tokens]
        return ", ".join(pairs)

    def run_inferencing(self):

        df= self.text_preprocessing()
        df["KPI"] = df["Sentence"].apply(self.extract_word_entity)
        return df

    
    def visualization(self, df):
        expanded_rows = []
        for _, row in df.iterrows():
            kpi_str = row["KPI"]
            if pd.isna(kpi_str) or kpi_str.strip() == "":
                continue
            # Split by comma to get individual pairs
            pairs = [p.strip() for p in kpi_str.split(",") if p.strip()]
            for pair in pairs:
                # Split on hyphen; assume format "word-entity", e.g. "discount-B-KPI"
                parts = pair.split("-")
                if len(parts) >= 2:
                    kpi_category = parts[-1]  # Extract the category (e.g., KPI, CY-VALUE, etc.)
                    expanded_rows.append({"KPI_Category": kpi_category})
        
        df_expanded = pd.DataFrame(expanded_rows)
        
        # Aggregate: count the number of occurrences per KPI category
        agg_df = df_expanded.groupby("KPI_Category").size().reset_index(name="Count")
        
        # Create a dummy root node to build a two-level hierarchy (root -> KPI Category)
        agg_df["Parent"] = "All KPI"
        
        # Create a dataframe for the root node
        root = pd.DataFrame({
            "KPI_Category": ["All KPI"],
            "Count": [agg_df["Count"].sum()],
            "Parent": [""]
        })
        
        # Combine root with the aggregated KPI categories
        sunburst_df = pd.concat([root, agg_df], ignore_index=True)
        
        # Create the sunburst chart
        fig = px.sunburst(
            sunburst_df,
            names="KPI_Category",
            parents="Parent",
            values="Count",
            color="KPI_Category",  # assign distinct colours per KPI category
            title="KPI Occurrences by Category",
            color_discrete_sequence=px.colors.qualitative.Plotly  # optional: choose a color palette
        )
        
        fig.show()






