"""
Team: Peter Zeng, Jihu Mun
File Description: This file describes the model and dataset classes for the 
Luar-based residual prediction model.

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
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset

class LuarResidualDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = self.tokenizer(item['text1'], item['text2'], return_tensors="pt", padding='max_length', truncation=True)
        # context = {key: val[0] for key, val in context.items()}  # Remove the extra batch dimension
        label = torch.tensor(item["residual"])
        return {"context": context, "labels": label}

class LuarResidual(torch.nn.Module):
    def __init__(self,**args):
        super().__init__()

        model_name = "rrivera1849/LUAR-MUD"
        self.loss_func = torch.nn.MSELoss(reduction = 'mean')
        self.enc_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)

        # Enhanced classifier with additional layers, dropout, and Tanh for the output
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(512, 256),  # First dense layer
            torch.nn.ReLU(),            # ReLU activation for non-linearity
            torch.nn.Dropout(0.1),      # Dropout for regularization
            torch.nn.Linear(256, 128),  # Second dense layer
            torch.nn.ReLU(),            # ReLU activation
            torch.nn.Dropout(0.1),      # Another dropout layer
            torch.nn.Linear(128, 1),    # Final layer to output the regression value
            torch.nn.Tanh()             # Tanh activation to ensure output is between -1 and 1
        )

    def forward(self, context, labels=None):
        # batch_size = context["input_ids"].size(0)
        # context["input_ids"] = context["input_ids"].reshape(batch_size, -1, 32)
        # context["attention_mask"] = context["attention_mask"].reshape(batch_size, -1, 32)
        outputs = self.enc_model(**context)
        # breakpoint()
        logits = self.classification_head(outputs)

        if labels is not None:
            loss = self.loss_func(logits.squeeze(-1), labels.float())
            return {
                "loss": loss,
                "logits": logits,
            }
        else:
            return {"logits": logits}