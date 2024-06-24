import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm.auto import tqdm
from explainable_module import Gram2VecModule
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
import argparse
import numpy as np
from residual_model.LongformerResidual import LongformerResidualDataset, LongformerResidual
from residual_model.RobertaResidual import RobertaResidualDataset, RobertaResidual
from residual_model.LuarResidual import LuarResidualDataset, LuarResidual
from residual_model.RobertaLargeResidual import RobertaLargeResidualDataset, RobertaLargeResidual
import matplotlib.pyplot as plt 
from transformers import AutoTokenizer
from peft import (LoraConfig, get_peft_model)
import csv
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
    parser.add_argument("-m", "--model_type", type=str, default="longformer", choices=["longformer", "roberta", "luar", "roberta-large", "style"], help="Type of model to use for training.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=["fanfiction", "reddit", "amazon"], help="Dataset to use for training.")
    parser.add_argument("-i", "--imbalanced", type=bool, default=False, help="Use imbalanced dataset.")
    parser.add_argument("-r", "--run_id", type=str, default="0", help="Run ID for the experiment.")
    parser.add_argument("-l", '--use_lora', type=bool, default = True,
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

    if args.model_type == "longformer":
        model = LongformerResidual()
        model_name = 'allenai/longformer-base-4096'
    elif args.model_type == "roberta":
        model = RobertaResidual()
        model_name = 'roberta-base'
    elif args.model_type == "roberta-large":
        model = RobertaLargeResidual()
        model_name = 'roberta-large'
    elif args.model_type == "luar":
        model = LuarResidual()
        model_name = 'rrivera1849/LUAR-MUD'
    else:
        raise ValueError("Unsupported model type specified.")

    if args.use_lora == True:
        print("using LORA")
        config = LoraConfig(
            r=args.LORA_R,
            lora_alpha=args.LORA_ALPHA,
            target_modules=args.LORA_TARGET_MODULES,
            lora_dropout=args.LORA_DROPOUT,
            bias="none",
        )
        model.enc_model = get_peft_model(model.enc_model, config)
        print(model.enc_model.print_trainable_parameters())

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.imbalanced == True:
        test_df = pd.read_csv(f"../data/imbalanced/{args.dataset}/test.csv")
    else:
        test_df = pd.read_csv(f"../data/{args.dataset}/test.csv")

    test_data = process_posts(test_df, g2v_vectorizer=g2v_vectorizer)
    # dev_data = process_posts(dev_df, g2v_vectorizer=g2v_vectorizer)

    if args.model_type == "longformer":
        test_dataset = LongformerResidualDataset(test_data, tokenizer)
        # dev_dataset = LongformerResidualDataset(dev_data, tokenizer)
    elif args.model_type == 'roberta':
        test_dataset = RobertaResidualDataset(test_data, tokenizer)
        # dev_dataset = RobertaResidualDataset(dev_data, tokenizer)
    elif args.model_type == 'roberta-large':
        test_dataset = RobertaLargeResidualDataset(test_data, tokenizer)
        # dev_dataset = RobertaLargeResidualDataset(dev_data, tokenizer)
    elif args.model_type == 'luar':
        test_dataset = LuarResidualDataset(test_data, tokenizer)
        # dev_dataset = LuarResidualDataset(dev_data, tokenizer)
    
    # model.load_state_dict(torch.load(f"../model/roberta_reddit_lora_model.pt"))
    if args.model_type == "roberta-large":
        if args.dataset == "reddit" or args.dataset == "fanfiction":
            model.load_state_dict(torch.load(f"../model/{args.model_type}_{args.dataset}_residual.pt")['model_state_dict'])
        else:
            model.load_state_dict(torch.load(f"../model/{args.model_type}_{args.dataset}_residual.pt"))

        # model.load_state_dict(torch.load(f"../model/roberta-large_amazon_model.pt"))
    else:
        # model.load_state_dict(torch.load(f"../model/{args.model_type}_{args.dataset}_{args.run_id}_residual.pt"))
        model.load_state_dict(torch.load(f"../model/{args.model_type}_{args.dataset}_residual.pt"))
        # model.load_state_dict(torch.load(f"../model/roberta_amazon_model.pt"))
    
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    gram2vec_cosims = []
    vector_map = pd.read_pickle(f'{args.dataset}_vector_map.pkl')

    for i, row in test_df.iterrows():
        vector1 = vector_map[row['post_1_id']]
        vector2 = vector_map[row['post_2_id']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = []

    model.eval()

    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="testing loop", position=1 ,leave=False)

        for batch in test_dataloader:
            # print(batch['context'])
            # breakpoint()
            with autocast():
                context = {k: v.to(device) for k, v in batch["context"].items()}
                labels = batch["labels"].to(device)  # Move labels to the device
                outputs = model(context, labels=labels)

            predictions = outputs["logits"].squeeze().detach().cpu().numpy()  # Adjust based on your model's output
            predicted_labels.append(predictions)

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]
    same_labels = list(test_df['same'])

    ### ANALYSIS CODE ###
    
    # test_data_for_csv = []
    # # Collect data for CSV
    # for i in range(len(predicted_labels)):
    #     test_data_for_csv.append([gram2vec_cosims[i], predicted_labels[i], residual_cosims[i], same_labels[i]])

    # Define the CSV file path
    # csv_file_path = f"final_testing/predictions_{args.model_type}_{args.dataset}.csv"

    # # Define the header for the CSV file
    # csv_header = ["gram2vec cosim", "predicted residual", "corrected cosim", 'same']

    # # Write the data to the CSV file
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(csv_header)
    #     for row in test_data_for_csv:
    #         writer.writerow(row)  # Add None for the threshold column

    # print(f"Test results for analysis saved to {csv_file_path}")

    # ### END ANALYSIS CODE ###

    ### EVALAUTION CODE ###
    import numpy as np

    def calculate_metrics(true_labels, predicted_labels):
        """
        Calculate precision, recall, and F1 score for a binary classification problem.
        
        Args:
        true_labels (np.array): Numpy array of true binary labels.
        predicted_labels (np.array): Numpy array of predicted binary labels.
        
        Returns:
        dict: A dictionary containing precision, recall, and F1 score for each class.
        """
        metrics = {}
        
        # Labels are assumed to be 0 and 1
        classes = [0, 1]
        
        for cls in classes:
            tp = np.sum((predicted_labels == cls) & (true_labels == cls))
            fp = np.sum((predicted_labels == cls) & (true_labels != cls))
            fn = np.sum((predicted_labels != cls) & (true_labels == cls))
            tn = np.sum((predicted_labels != cls) & (true_labels != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(cls, precision, recall, f1)
            metrics[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        return metrics
    
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in thresholds:
        g2v_correct = 0
        long_correct = 0
        true_labels = list(test_df['same'])

        for i in range(len(true_labels)):
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
        # print(f"threshold: {threshold}, gram2vec correct: {g2v_correct}/{total}, residual correct: {long_correct}/{total}")
        # print(f"gram2vec accuracy: {g2v_correct/len(true_labels):.4f}, residual accuracy: {long_correct/len(true_labels):.4f}")
        from sklearn.metrics import precision_score, recall_score, f1_score

        # Convert lists to numpy arrays for compatibility with sklearn metrics
        true_labels_np = np.array(true_labels)
        predicted_labels_residual = np.array([1 if score > threshold else 0 for score in residual_cosims])
        # print(calculate_metrics(true_labels_np, predicted_labels_residual))
        predicted_labels_gram2vec = np.array([1 if score > threshold else 0 for score in gram2vec_cosims])

        gram2vec_f1_per_class = f1_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        gram2vec_precision_per_class = precision_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        gram2vec_recall_per_class = recall_score(true_labels_np, predicted_labels_gram2vec, average=None, zero_division=0)
        
        f1_per_class = f1_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        precision_per_class = precision_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels_np, predicted_labels_residual, average=None, zero_division=0)
        # print("Gram2Vec Stats:")
        # print(f"Different Author Precision: {gram2vec_precision_per_class[0]:.4f}, Same Author Precision: {gram2vec_precision_per_class[1]:.4f}")
        # print(f"Different Author Recall: {gram2vec_recall_per_class[0]:.4f}, Same Author Recall: {gram2vec_recall_per_class[1]:.4f}")
        # print(f"Different Author F1: {gram2vec_f1_per_class[0]:.4f}, Same Author F1: {gram2vec_f1_per_class[1]:.4f}")
        # print("Residual Stats:")
        # print(f"Different Author Precision: {precision_per_class[0]:.4f}, Same Author Precision: {precision_per_class[1]:.4f}")
        # print(f"Different Author Recall: {recall_per_class[0]:.4f}, Same Author Recall: {recall_per_class[1]:.4f}")
        # print(f"Different Author F1: {f1_per_class[0]:.4f}, Same Author F1: {f1_per_class[1]:.4f}")
        # print()

        # Define the CSV file path
        if args.imbalanced:
            csv_file_path = f"imbalanced/results_{args.model_type}_{args.dataset}_{args.run_id}.csv"
        else: 
            csv_file_path = f"new_results/residual/residual_results_{args.model_type}_{args.dataset}_{args.run_id}.csv"

        # # Define the header for the CSV file
        csv_header = [
            "threshold", "gram2vec_correct/total", "residual_correct/total", "gram2vec_same_author_f1",  
            "residual_same_author_f1", "gram2vec_accuracy", "residual_accuracy",
            "residual_same_author_precision", "residual_same_author_recall", 
            "residual_diff_author_precision", "residual_diff_author_recall", "residual_diff_author_f1",
            "gram2vec_same_author_precision", "gram2vec_same_author_recall",
            "gram2vec_diff_author_precision", "gram2vec_diff_author_recall", "gram2vec_diff_author_f1"
        ]

        # Calculate the ROC curve and AUC for residual_cosims
        fpr_residual, tpr_residual, _ = roc_curve(true_labels_np, residual_cosims)
        auc_residual = auc(fpr_residual, tpr_residual)

        # Calculate the ROC curve and AUC for gram2vec_cosims
        fpr_gram2vec, tpr_gram2vec, _ = roc_curve(true_labels_np, gram2vec_cosims)
        auc_gram2vec = auc(fpr_gram2vec, tpr_gram2vec)

        # print(f"gram2vec AUC: {auc_gram2vec:.3f}, residual AUC: {auc_residual:.3f}")
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr_residual, tpr_residual, color='darkorange', lw=2, label=f'Residual AUC = {auc_residual:.3f}')
        plt.plot(fpr_gram2vec, tpr_gram2vec, color='blue', lw=2, label=f'Gram2Vec AUC = {auc_gram2vec:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUC Curve for {args.model_type} on {args.dataset}')
        plt.legend(loc="lower right")
        plt.grid()
        if args.imbalanced:
            plt.savefig(f"imbalanced/graphs/{args.model_type}_{args.dataset}_{args.run_id}_residual_auc.png")
        else:
            plt.savefig(f"new_results/residual/graphs/{args.model_type}_{args.dataset}_{args.run_id}_residual_auc.png")

        # # Check if the CSV file already exists
        file_exists = os.path.isfile(csv_file_path)

        # # Open the CSV file in append mode
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the header only if the file does not exist
            if not file_exists:
                writer.writerow(csv_header)

            # Write the stats line to the CSV file
            writer.writerow([
                threshold, f"{g2v_correct}/{total}", f"{long_correct}/{total}", 
                f"{gram2vec_f1_per_class[1]:.4f}", f"{f1_per_class[1]:.4f}",
                f"{g2v_correct/len(true_labels):.4f}", f"{long_correct/len(true_labels):.4f}",
                f"{precision_per_class[1]:.4f}", f"{recall_per_class[1]:.4f}",
                f"{precision_per_class[0]:.4f}", f"{recall_per_class[0]:.4f}", f"{f1_per_class[0]:.4f}",
                f"{gram2vec_precision_per_class[1]:.4f}", f"{gram2vec_recall_per_class[1]:.4f}",
                f"{gram2vec_precision_per_class[0]:.4f}", f"{gram2vec_recall_per_class[0]:.4f}",
                f"{gram2vec_f1_per_class[0]:.4f}"
            ])
    ### END EVALAUTION CODE ###

    # Plotting the residuals
    # plt.figure(figsize=(10, 6))
    # plt.hist(predicted_labels, bins=50, color='skyblue', edgecolor='black')
    # plt.title('predicted residuals in test')
    # plt.xlabel('Residual Value')
    # plt.ylabel('Frequency')
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()
    # plt.savefig(f"results/dev_predicted_labels_{args.model_type}_{args.dataset}.png")

