import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset

class ResidualDataset(Dataset):
    def __init__(self, data, model_type):
        self.data = data
        self.model_type = model_type
        if model_type == "longformer":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        elif model_type == "roberta":
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        elif model_type == "roberta-large":
            self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        elif model_type == "luar":
            self.tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD")

        self.max_length = 1024 if model_type == 'longformer' else 512
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = self.tokenizer(item['text1'], item['text2'], return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        context = {key: val[0] for key, val in context.items()}  # Remove the extra batch dimension
        label = torch.tensor(item["residual"])
        
        return {"context": context, "labels": label}

class ResidualModel(torch.nn.Module):
    def __init__(self, model_type, **args):
        super().__init__()
        
        self.model_type = model_type
        self.loss_func = torch.nn.MSELoss(reduction='mean')
        
        if model_type == 'longformer':
            model_name = "allenai/longformer-base-4096"
            hidden_size = 768
        elif model_type == 'roberta':
            model_name = "roberta-base"
            hidden_size = 768
        elif model_type == 'roberta-large':
            model_name = "roberta-large"
            hidden_size = 1024
        elif model_type == 'luar':
            model_name = "rrivera1849/LUAR-MUD"
            hidden_size = 512
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.enc_model = AutoModel.from_pretrained(model_name, trust_remote_code=(model_type == 'luar'))
        self.enc_tok = AutoTokenizer.from_pretrained(model_name)

        self.classification_head = self._create_classification_head(hidden_size)

    def _create_classification_head(self, hidden_size):
        return torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 4, 1),
            torch.nn.Tanh()
        )

    def forward(self, context, labels=None):
        if self.model_type in ['roberta', 'roberta-large', 'longformer']:
            outputs = self.enc_model(**context, output_hidden_states=True).hidden_states[-1]
            embeds = outputs[:, 0, :]
        elif self.model_type == 'luar':
            outputs = self.enc_model(**context)
            embeds = outputs

        logits = self.classification_head(embeds)

        if labels is not None:
            loss = self.loss_func(logits.squeeze(-1), labels.float())
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}