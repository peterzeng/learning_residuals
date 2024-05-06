"""
Team: Peter Zeng, Jihu Mun
File Description: This file describes the model and dataset classes for the 
Longformer-based residual prediction model.

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

class LongformerResidualDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = 2048
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = self.tokenizer(item['text1'], item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        context = {key: val[0] for key, val in context.items()}  # Remove the extra batch dimension
        label = torch.tensor(item["residual"])
        return {"context": context, "labels": label}

class LongformerResidual(torch.nn.Module):
    def __init__(self,**args):
        super().__init__()

        model_name = "allenai/longformer-base-4096"
        self.loss_func = torch.nn.MSELoss(reduction = 'mean')
        self.enc_model = AutoModel.from_pretrained(model_name)
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)

        # Enhanced classifier with additional layers, dropout, and Tanh for the output
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(768, 512),  # First dense layer
            torch.nn.ReLU(),            # ReLU activation for non-linearity
            torch.nn.Dropout(0.1),      # Dropout for regularization
            torch.nn.Linear(512, 256),  # Second dense layer
            torch.nn.ReLU(),            # ReLU activation
            torch.nn.Dropout(0.1),      # Another dropout layer
            torch.nn.Linear(256, 1),    # Final layer to output the regression value
            torch.nn.Tanh()             # Tanh activation to ensure output is between -1 and 1
        )

    def forward(self, context, labels=None):
        outputs = self.enc_model(input_ids=context["input_ids"],
                                 attention_mask=context["attention_mask"],
                                 output_hidden_states=True
                                ).hidden_states[-1]

        indices = [max(loc for loc, val in enumerate(c.tolist()) if val ==  2) for c in context["input_ids"]]

        # Gather the embeddings corresponding to the SEP token
        embeds = torch.stack([t[sep] for t, sep in zip(outputs, indices)])

        # Pass the SEP embeddings through the classification head
        logits = self.classification_head(embeds)

        # Calculate loss if labels are provided
        if labels is not None:
            loss = self.loss_func(logits.squeeze(-1), labels.float())
            return {
                "loss": loss,
                "logits": logits,
            }
        else:
            return {"logits": logits}
        
# class LongformerResidual(torch.nn.Module):
#     def __init__(self,**args):
#         super().__init__()

#         model_name = "allenai/longformer-base-4096"
#         self.loss_func = torch.nn.MSELoss(reduction = 'mean')
#         self.enc_model = AutoModel.from_pretrained(model_name)
#         self.enc_tok = AutoTokenizer.from_pretrained(model_name)

#         # Define the linear classifier
#         # dense layer linear/relu/linear
#         # self.classification_head = torch.nn.Sequential(
#         #     torch.nn.Linear(
#         #         768, 768
#         #     ),  # Dense projection layer.
#         #     torch.nn.Tanh(),  # Activation. TODO: Dropout?
#         #     torch.nn.Linear(768, 1)  # Classifier.
#         # )
#         self.linear_classifier = torch.nn.Linear(768, 1)  # Adjust the input size according to the output size of the Style-Embedding model

#         # Add a dropout layer
#         # self.dropout = torch.nn.Dropout(0.3)

#     def forward(self, context, labels=None):
#         # Reshape the input tensors
#         outputs = self.enc_model(input_ids=context["input_ids"],
#                     attention_mask=context["attention_mask"],
#                      output_hidden_states=True
#                     ).hidden_states[-1]

#         indices = [max(loc for loc, val in enumerate(c.tolist()) if val ==  2) for c in context["input_ids"]]

#         #stack the SEP embeddings of the entire batch into a tensor
#         # gather the embeddings corresponding to the SEP token
#         # used to mark the end of a segment within the input sequence
#         embeds = torch.stack([t[sep] for t, sep in zip(outputs, indices)])

#         # Classify and calculate loss
#         activation_function = torch.nn.Tanh()
#         logits = self.linear_classifier(embeds)
#         logits = activation_function(logits)
#         # logits = torch.clamp(logits, min=-1, max=2)
#         # labels = labels.unsqueeze(1) # [[number]]
#         # print(labels.shape)

#         # loss = self.loss_func(logits, labels.float())
#         loss = self.loss_func(logits.squeeze(-1), labels.float())

#         return {
#             "loss": loss,
#             "logits": logits,
#         }