"""
Team: Peter Zeng, Jihu Mun
File Description: This file takes the binary classification finetuned models, 
and task finetunes them on the residual prediction task.

NLP Topics criteria summary: We include this in every file for visibility.
I. Classification: We get a baseline of the models' performance 
without using residuals by task finetuning them on binary classification for authorship verification.
This occurs in baseline.py

II. Semantics: In order to train our residual prediction model, 
we first get the embeddings of pairs of Reddit posts by tokenizing them, 
passing them through the respective model, and taking the last hidden layer.
This occurs in train_residual.py and finetune_baseline.py

III. Language Modeling: We use autoencoder models for our sequence classification/regression tasks.
This occurs in all of our files. 

IV. Applications: We apply the residual prediction model for the task of authorship verification.
This occurs in our training and testing files.

System: All experiments were conducted on an A6000 GPU.
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.utils.data import Dataset
import argparse
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()  # Initialize the gradient scaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AuthorClassificationModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        if model_name == "luar":
            self.model = AutoModel.from_pretrained('rrivera1849/LUAR-MUD', trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained('rrivera1849/LUAR-MUD')
        elif model_name == "longformer":
            self.model = AutoModel.from_pretrained('allenai/longformer-base-4096')
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
        elif model_name == "roberta":
            self.model = AutoModel.from_pretrained('roberta-base')
            self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        else:
            raise ValueError("Unsupported model type specified.")
        
        if model_name == "luar":
            self.classifier = torch.nn.Linear(512, 2)
        elif model_name == "longformer" or model_name == "roberta":
            self.classifier = torch.nn.Linear(768, 2)
        else:
            raise ValueError("Unsupported model type specified.")
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        if self.model_name != "luar":
            sequence_output = outputs[0]  # Get the sequence output
            pooled_output = sequence_output[:, 0]  # Use the first token's embeddings (like [CLS] token in BERT)
            logits = self.classifier(pooled_output)  # Pass through the classifier
        else:
            logits = self.classifier(outputs)
        # print(outputs.shape)
        # logits = self.classifier(outputs)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()  # Loss function for classification
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return {
                "loss": loss,
                "logits": logits
            }
        return logits

def process_posts(post_df):
    data = []
    for i, row in tqdm(post_df.iterrows(), total=post_df.shape[0], desc="Processing Posts"):
        data.append({"text1":row['post_1'],
                        "text2":row['post_2'],
                        "label":int(row['same'])})
    
    return data

class DocumentPairDataset(Dataset):
    def __init__(self, data, tokenizer, model_name):
        self.data = data
        self.tokenizer = tokenizer
        self.model_name = model_name
        if model_name == "longformer":
            self.max_length = 1024
    
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.model_name == "longformer":
            context = self.tokenizer(item['text1'], item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        else:
            context = self.tokenizer(item['text1'], item['text2'], return_tensors="pt", padding='max_length', truncation=True)
        
        if self.model_name != "luar":
            context = {key: val[0] for key, val in context.items()}  # Remove the extra batch dimension
        # tokenized_batch['input_ids'] = tokenized_batch['input_ids'].reshape(actual_batch_size, 512, -1).to(device)
        # tokenized_batch['attention_mask'] = tokenized_batch['attention_mask'].reshape(actual_batch_size, 512, -1).to(device)
        label = torch.tensor(item["label"])
        return {"context": context, "labels": label}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on residual data.")
    parser.add_argument("-m", "--model_type", type=str, default="longformer", choices=["longformer", "roberta", "luar"], help="Type of model to use for training.")
    parser.add_argument("-r", "--run_id", type=str, default="0", help="Run ID for the experiment.")
    parser.add_argument("-p", "--percentage", type=float, default=0.2, help="Percentage of data to sample for training, validation, and testing.")
    parser.add_argument('--use_lora', type=bool, default = False,
                    help='Lora rank.')
    parser.add_argument('--LORA_R', type=int, default = 8,
                    help='Lora rank.')
    parser.add_argument('--LORA_ALPHA', type=int, default = 16,
                    help='Lora alpha')
    parser.add_argument('--LORA_DROPOUT', type=float, default = 0.05,
                    help='Lora dropout')
    parser.add_argument('--LORA_TARGET_MODULES', nargs='+', default=["query","value",], 
                    help='List of LORA target modules')
    args = parser.parse_args()

    if args.model_type == "longformer":
        model_name = 'allenai/longformer-base-4096'
    elif args.model_type == "roberta":
        model_name = 'roberta-base'
    elif args.model_type == "luar":
        model_name = 'rrivera1849/LUAR-MUD'
    else:
        raise ValueError("Unsupported model type specified.")

    model = AuthorClassificationModel(args.model_type)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = pd.read_csv("../data/train.csv")
    dev_df = pd.read_csv("../data/dev.csv")
    test_df = pd.read_csv("../data/test.csv")

    train_df_sampled = train_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
    dev_df_sampled = dev_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
    test_df_sampled = test_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility

    train_data = process_posts(train_df_sampled)
    dev_data = process_posts(dev_df_sampled)
    test_data = process_posts(test_df_sampled)
    
    train_dataset = DocumentPairDataset(train_data, tokenizer, args.model_type)
    dev_dataset = DocumentPairDataset(dev_data, tokenizer, args.model_type)
    test_dataset = DocumentPairDataset(test_data, tokenizer, args.model_type)

    if args.model_type == 'longformer':
        b_size = 8
        print(f"Using batch size of {b_size}")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)
    else:
        b_size = 32
        print(f"Using batch size of {b_size}")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    if args.use_lora == True:
        print("Using LORA")
        config = LoraConfig(
            r=args.LORA_R,
            lora_alpha=args.LORA_ALPHA,
            target_modules=args.LORA_TARGET_MODULES,
            lora_dropout=args.LORA_DROPOUT,
            bias="none",
        )
        model.model = get_peft_model(model.model, config)
        print(model.model.print_trainable_parameters())
        # print("Checking number of params", model.enc_model.print_trainable_parameters())

    model.to(device)
    num_epochs = 10
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    best_val_loss = float('inf')  # Initialize the best validation loss to infinity
    early_stopping_counter = 0
    early_stopping_patience = 3
    
    accumulation_steps = 1 # Adjust this based on your GPU capacity and desired batch size
    print(f"number of accuulation steps: {accumulation_steps}")

    for epoch in range(num_epochs):
        model.train()

        train_dataloader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", position=1, leave=True)
        optimizer.zero_grad()
        train_loss = 0

        for i, batch in enumerate(train_dataloader):
            input_ids = batch['context']['input_ids'].to(device)
            attention_mask = batch['context']['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]  
                loss = loss / accumulation_steps  # Normalize loss to account for accumulation
            # loss.backward()
            scaler.scale(loss).backward()  # Scale the loss and call backward() to create scaled gradients

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Reset gradients after updating

            train_loss += loss.item() * accumulation_steps  # Scale up loss to its true value

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}")

        dev_dataloader = tqdm(dev_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", position=1, leave=True)
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids = batch['context']['input_ids'].to(device)
                attention_mask = batch['context']['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                val_loss += loss.item()

        avg_val_loss = val_loss / len(dev_dataloader)
        print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}")

        # Check if the current validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model
            torch.save(model.state_dict(), f'../model/{args.model_type}_{args.run_id}_baseline.pt')
            print(f"New best validation loss: {best_val_loss}. Model saved!")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Testing loop with a summary of per class F1, precision, and recall
    model.eval()
    all_labels = []
    all_predictions = []
    test_dataloader = tqdm(test_dataloader, desc=f"Testing", position=1, leave=True)
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['context']['input_ids'].to(device)
            attention_mask = batch['context']['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs
            predictions = torch.argmax(logits, dim=-1)
            labels = batch['labels'].to(device)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # Calculate metrics per class
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    # Calculate precision, recall, f1, and support for each class without averaging
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)

    # Print per class results
    print("Class-wise metrics:")
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}")

    # Calculate and print total metrics using 'macro' average
    total_precision, total_recall, total_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)
    print(f"Total Precision: {total_precision:.4f}")
    print(f"Total Recall: {total_recall:.4f}")
    print(f"Total F1 Score: {total_f1:.4f}")

    # Calculate and print total accuracy
    total_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Total Accuracy: {total_accuracy:.4f}")

    # Calculate and print accuracy for each class
    unique_labels = np.unique(all_labels)
    for label in unique_labels:
        class_mask = (all_labels == label)
        class_accuracy = accuracy_score(np.array(all_labels)[class_mask], np.array(all_predictions)[class_mask])
        print(f"Class {label} Accuracy: {class_accuracy:.4f}")