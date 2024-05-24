import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LongformerModel, LongformerTokenizer
import torch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from torch.optim import AdamW
from tqdm.auto import tqdm
from residual_model.LongformerResidual import LongformerResidualDataset, LongformerResidual
from residual_model.RobertaResidual import RobertaResidualDataset, RobertaResidual
from residual_model.LuarResidual import LuarResidualDataset, LuarResidual
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from explainable_module import Gram2VecModule
from tqdm.auto import tqdm
import argparse
import numpy as np
from transformers import logging
import matplotlib.pyplot as plt
from data_utils import get_model_and_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    # get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    # set_peft_model_state_dict,
)
from peft import PeftModel, PeftConfig


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

logging.set_verbosity(logging.ERROR)

def process_posts(post_df, g2v_vectorizer):
    data = []
    for i, row in tqdm(post_df.iterrows(), total=post_df.shape[0], desc="Processing Posts"):
        residual = row['same'] - g2v_vectorizer.get_vector_and_score(row['post_1'], row['post_2'], row['post_1_id'], row['post_2_id'])
        data.append({"text1":row['post_1'],
                        "text2":row['post_2'],
                        "residual":residual})
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on residual data.")
    parser.add_argument("-m", "--model_type", type=str, default="longformer", choices=["longformer", "roberta", "luar"], help="Type of model to use for training.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=["fanfiction", "reddit"], help="Dataset to use for training.")
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

    g2v_vectorizer = Gram2VecModule(dataset=args.dataset)

    vector_map = pd.read_pickle('vector_map.pkl')
    test_df = pd.read_csv("../data/test.csv")

    test_df_sampled = test_df.sample(frac=args.percentage, random_state=42)
    test_data = process_posts(test_df_sampled)

    model, test_dataset = get_model_and_dataset(args.model_type, test_data)
    # print(model)
    # model.load_state_dict(torch.load(f"../model/model.pt"))
    if args.use_lora == True:
        config = LoraConfig(
            r=args.LORA_R,
            lora_alpha=args.LORA_ALPHA,
            target_modules=args.LORA_TARGET_MODULES,
            lora_dropout=args.LORA_DROPOUT,
            bias="none",
        )
        print(f"Using PEFT, trying to load {args.model_type}_{args.run_id}_model.pt")
        # model = PeftModel(model, config)    
        model.enc_model = get_peft_model(model.enc_model, config)

    model.load_state_dict(torch.load(f"../model/{args.model_type}_{args.run_id}_model.pt"))
    model.to(device)

    model.eval()

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    gram2vec_cosims = []

    for i, row in test_df_sampled.iterrows():
        vector1 = vector_map[row['post_1_id']]
        vector2 = vector_map[row['post_2_id']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = []

    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test loop", position=1 ,leave=False)

        for batch in test_dataloader:
            # print(batch['context'])
            # breakpoint()
            context = {k: v.to(device) for k, v in batch["context"].items()}
            labels = batch["labels"].to(device)  # Move labels to the device
            outputs = model(context, labels=labels)
            # print(context)
            # print(outputs)        
            # Store predictions and labels to compute MAE and Pearson r
            # print(outputs["logits"])
            predictions = outputs["logits"].squeeze().detach().cpu().numpy()  # Adjust based on your model's output
            predicted_labels.append(predictions)

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]

    thresholds = [0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
    for threshold in thresholds:
        g2v_correct = 0
        long_correct = 0
        true_labels = list(test_df_sampled['same'])

        for i in range(len(true_labels)):
            # if i % 20 == 0:
                # print(gram2vec_cosims[i], longformer_cosims[i])
            # if longformer_cosims[i] < 0:
                # print(longformer_cosims[i])
            if true_labels[i] == 1:
                if (gram2vec_cosims[i] > threshold):
                    g2v_correct += 1
                if (residual_cosims[i] > threshold):
                    long_correct += 1
                    
            elif true_labels[i] == 0:
                if (gram2vec_cosims[i] < threshold):
                    g2v_correct += 1
                if (residual_cosims[i] < threshold):
                    long_correct += 1

        total = len(true_labels)
        print(f"threshold: {threshold}, gram2vec correct: {g2v_correct}/{total}, residual correct: {long_correct}/{total}")
        print(f"gram2vec accuracy: {g2v_correct/len(true_labels):.4f}, residual accuracy: {long_correct/len(true_labels):.4f}")
        from sklearn.metrics import precision_score, recall_score, f1_score

        # Convert lists to numpy arrays for compatibility with sklearn metrics
        true_labels_np = np.array(true_labels)
        predicted_labels_residual = np.array([1 if score > threshold else 0 for score in residual_cosims])
        predicted_labels_gram2vec = np.array([1 if score > threshold else 0 for score in gram2vec_cosims])

        gram2vec_f1_per_class = f1_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        gram2vec_precision_per_class = precision_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        gram2vec_recall_per_class = recall_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        
        f1_per_class = f1_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        precision_per_class = precision_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        print("Gram2Vec Stats:")
        print(f"Different Author Precision: {gram2vec_precision_per_class[0]:.4f}, Same Author Precision: {gram2vec_precision_per_class[1]:.4f}")
        print(f"Different Author Recall: {gram2vec_recall_per_class[0]:.4f}, Same Author Recall: {gram2vec_recall_per_class[1]:.4f}")
        print(f"Different Author F1: {gram2vec_f1_per_class[0]:.4f}, Same Author F1: {gram2vec_f1_per_class[1]:.4f}")
        print("Residual Stats:")
        print(f"Different Author Precision: {precision_per_class[0]:.4f}, Same Author Precision: {precision_per_class[1]:.4f}")
        print(f"Different Author Recall: {recall_per_class[0]:.4f}, Same Author Recall: {recall_per_class[1]:.4f}")
        print(f"Different Author F1: {f1_per_class[0]:.4f}, Same Author F1: {f1_per_class[1]:.4f}")
        print()

    # Plotting the residuals
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_labels, bins=50, color='skyblue', edgecolor='black')
    plt.title('predicted residuals in test')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    plt.savefig(f"predicted_labels.png")