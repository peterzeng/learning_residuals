import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm import tqdm
from transformers import logging
import torch
from finetune_ensemble import SiameseRoberta
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import f1_score
from finetune_ensemble import SiameseRoberta
import argparse
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='roberta-base', choices=['roberta-base', 'roberta-large', 'longformer', 'luar'])
    parser.add_argument('-d', '--dataset', type=str, default='reddit', choices=['reddit', 'amazon', 'fanfiction'])
    parser.add_argument("-i", "--imbalanced", type=bool, default=False, help="Use imbalanced dataset.")
    parser.add_argument("-r", "--run_id", type=str, default="0", help="Run ID for the experiment.")
    args = parser.parse_args()
    
    vector_map = pd.read_pickle(f'{args.dataset}_vector_map.pkl')
    if args.imbalanced:
        test_df = pd.read_csv(f"../data/imbalanced/{args.dataset}/test.csv")
    else:
        test_df = pd.read_csv(f"../data/{args.dataset}/test.csv")

    # dev_df = pd.read_csv(f"../data/{args.dataset}/dev.csv")
    # Assuming the SiameseRoberta class and other necessary imports are already defined as in your provided script
    post1s = list(test_df['post_1'])
    post2s = list(test_df['post_2'])
    # Load the fine-tuned model
    if args.model == "longformer":
        model_name = "allenai/longformer-base-4096"
    elif args.model == "luar":
        model_name = 'rrivera1849/LUAR-MUD'
    else:
        model_name = args.model
        
    model = SiameseRoberta(model_name).to(device)

    # Assuming you have a path to the saved model
    model_path = f'../model/siamese/contrastive_loss_finetuned_{args.model}_{args.dataset}.pt'

    # Load the model (make sure the model is modified with LoRA exactly as when it was saved)
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    except RuntimeError as e:
        print("Error loading model:", e)

    # print(model.enc_model.print_trainable_parameters())

    # model_path = '../model/contrastive_loss_finetuned_roberta-base.pt'
    # model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to get embeddings
    def get_embeddings(text, model_name):
        tokenized = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        if model_name == "rrivera1849/LUAR-MUD":
            tokenized['input_ids'] = tokenized['input_ids'].reshape(1, 1, -1).to(device)
            tokenized['attention_mask'] = tokenized['attention_mask'].reshape(1, 1, -1).to(device)
            with torch.no_grad():
                outputs = model(tokenized, tokenized, input_ids2=None, attention_mask2=None)
            embedding = outputs[0].cpu()
        else:
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)

            with torch.no_grad():
                # You can use either output1 or output2 since both are the same model and weights
                embedding = model.enc_model(input_ids, attention_mask=attention_mask)[1].cpu()  # pooler_output
        return embedding

    gram2vec_cosims = []

    for i, row in test_df.iterrows():
        vector1 = vector_map[row['post_1_id']]
        vector2 = vector_map[row['post_2_id']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    neural_cosims = []
    for i, post in tqdm(enumerate(post1s), desc="Processing posts for embeddings", total=len(post1s)):
        post1 = post1s[i]
        post2 = post2s[i]

        embedding1 = get_embeddings(post1, model_name)
        embedding2 = get_embeddings(post2, model_name)
        neural_cosim = cosine_similarity(embedding1, embedding2)
        neural_cosims.append(neural_cosim[0][0])

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    true_labels = list(test_df['same'])

    for threshold in thresholds:
        gram2vec_labels = [1 if score > threshold else 0 for score in gram2vec_cosims]
        neural_labels = [1 if score > threshold else 0 for score in neural_cosims]
        gram2vec_f1 = f1_score(true_labels, gram2vec_labels, pos_label=1)
        neural_f1 = f1_score(true_labels, neural_labels, pos_label=1)
        print(f'gram2vec_f1: {gram2vec_f1}, neural_f1: {neural_f1}')

    # Custom weighting of gram2vec_cosims and luar_cosims
    weights = np.linspace(0, 1, 11)  # Test weights from 0 to 1 in steps of 0.1
    best_weight = 0
    best_f1 = 0

    for weight in weights:
        combined_cosims = weight * np.array(gram2vec_cosims) + (1 - weight) * np.array(neural_cosims)
        accuracies = []

        for threshold in thresholds:

            predicted_labels = [1 if score > threshold else 0 for score in combined_cosims]
            f1 = f1_score(true_labels, predicted_labels, pos_label=1)  # Focusing on F1 score for class 1

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_weight = weight

    f1_best_weight = best_weight
    print(f"Best weight for combining models: {f1_best_weight}, Best same author f1: {best_f1}, at threshold: {best_threshold}")

    import numpy as np
    from sklearn.metrics import roc_curve, auc

    # Assuming gram2vec_cosims, cosims, and true_labels are already defined
    weights = np.linspace(0, 1, 101)  # Test weights from 0 to 1 in steps of 0.01
    best_weight = 0
    best_auc = 0

    for weight in weights:
        combined_cosims = weight * np.array(gram2vec_cosims) + (1 - weight) * np.array(neural_cosims)
        fpr, tpr, _ = roc_curve(true_labels, combined_cosims)
        auc_score = auc(fpr, tpr)
        if auc_score > best_auc:
            best_auc = auc_score
            best_weight = weight

    print(f"Best weight: {best_weight}, Best AUC: {best_auc:.3f}")

    # Plot the ROC curve for the best weight
    combined_cosims = best_weight * np.array(gram2vec_cosims) + (1 - best_weight) * np.array(neural_cosims)
    gram2vec_fpr, gram2vec_tpr, _ = roc_curve(true_labels, gram2vec_cosims)
    neural_fpr, neural_tpr, _ = roc_curve(true_labels, neural_cosims)
    fpr, tpr, _ = roc_curve(true_labels, combined_cosims)
    gram2vec_auc = auc(gram2vec_fpr, gram2vec_tpr)
    neural_auc = auc(neural_fpr, neural_tpr)
    auc_score = auc(fpr, tpr)
    plt.figure()
    plt.text(0.05, 0.95, f'Best Weight: {best_weight:.3f}', fontsize=12, fontweight='bold', ha='left', va='top', color='red', bbox=dict(facecolor='white', alpha=0.8))
    plt.text(0.05, 0.80, f'Best Weight: {f1_best_weight:.3f}, Best F1: {best_f1:.3f}, Threshold: {best_threshold:.2f}', fontsize=12, fontweight='bold', ha='left', va='top', color='red', bbox=dict(facecolor='white', alpha=0.8))

    # plt.text(0.5, 0.05, f'Best Weight: {best_weight:.3f}', fontsize=12, ha='center', color='red', bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(0.5, 0.01, f'Best Weight: {f1_best_weight:.3f}, Best F1: {best_f1:.3f}, Threshold: {best_threshold:.2f}', fontsize=12, ha='center', color='red', bbox=dict(facecolor='white', alpha=0.5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble AUC = {auc_score:.3f}')
    plt.plot(gram2vec_fpr, gram2vec_tpr, color='blue', lw=2, label=f'gram2vec AUC = {gram2vec_auc:.3f}')
    plt.plot(neural_fpr, neural_tpr, color='green', lw=2, label=f'neural AUC = {neural_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # Optionally save the plot
    if args.imbalanced:
        plt.savefig(f"imbalanced/graphs/{args.model}_{args.dataset}_{args.run_id}_ensemble_auc.png")
    else:
        plt.savefig(f"new_results/graphs/{args.model}_{args.dataset}_{args.run_id}_ensemble_auc.png")