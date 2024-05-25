'''
    This code trains a model to predict the residual 
    between ground truth and predicted cosine similarities from interpretable system.
    It outputs 
'''
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from residual_model.LongformerResidual import LongformerResidualDataset, LongformerResidual
from residual_model.RobertaResidual import RobertaResidualDataset, RobertaResidual
from residual_model.LuarResidual import LuarResidualDataset, LuarResidual
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from explainable_module import Gram2VecModule
from torch.cuda.amp import autocast
import logging
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os
import csv

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
    parser.add_argument("-m", "--model_type", type=str, default="longformer", choices=["longformer", "roberta", "luar"], help="Type of model to use for training.")
    parser.add_argument("-d", "--dataset", type=str, default="reddit", choices=["fanfiction", "reddit", "amazon"], help="Dataset to use for training.")
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

    #use qlora
    parser.add_argument('--use_qlora', type=bool, default = False,
                    help='Use QLoRA.')
    #BitsAndBytesConfig
    parser.add_argument('--use_qlora', type=bool, default = False,
                    help='Use QLoRA.')
    parser.add_argument('--load_in_8bit', type=bool, default = True,
                    help='load in 8 bit')
    parser.add_argument('--load_in_4bit', type=bool, default = False,
                    help='load in 4 bit.')
    parser.add_argument('--llm_int8_threshold', type=int, default = 6.0,
                    help='llm_int8_threshold.')
    parser.add_argument('--llm_int8_has_fp16_weight', type=bool, default = False,
                    help='llm_int8_has_fp16_weight.')
    parser.add_argument('--bnb_4bit_use_double_quant', type=bool, default = True,
                    help='bnb_4bit_use_double_quant.')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default = "nf4",
                    help='bnb_4bit_quant_type.')
    
    #setting device map for qlora settings, might not be required -> need further testing
    device_map = "auto"
    # if we are in a distributed setting, we need to set the device map and max memory(iffy) per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    print(device_map)


    args = parser.parse_args()

    g2v_vectorizer = Gram2VecModule(dataset=args.dataset)

    if args.model_type == "longformer":
        model = LongformerResidual()
        model_name = 'allenai/longformer-base-4096'
    elif args.model_type == "roberta":
        model = RobertaResidual()
        model_name = 'roberta-base'
    elif args.model_type == "luar":
        model = LuarResidual()
        model_name = 'rrivera1849/LUAR-MUD'
    else:
        raise ValueError("Unsupported model type specified.")
    
    if args.use_qlora == True:
        print("Using Qlora")
        qlora_config = BitsAndBytesConfig(
                # load_in_4bit=True,
                load_in_8bit=args.load_in_8bit,
                llm_int8_threshold=args.llm_int8_threshold,
                llm_int8_has_fp16_weight=args.llm_int8_has_fp16_weight,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            )
        model.enc_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=args.load_in_8bit,
            device_map=device_map,
            # max_memory=max_memory,
            quantization_config=qlora_config,
            torch_dtype=torch.bfloat16,
        )
        model.enc_model = prepare_model_for_kbit_training(model.enc_model, use_gradient_checkpointing=True)
        #quantization done, if quantization -> set use_lora to true as well
        args.use_lora = True

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

    train_df = pd.read_csv(f"../data/{args.dataset}/train.csv")
    dev_df = pd.read_csv(f"../data/{args.dataset}/dev.csv")
    test_df = pd.read_csv(f"../data/{args.dataset}/test.csv")

    if args.percentage != 1:
        train_df_sampled = train_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
        dev_df_sampled = dev_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility
        test_df_sampled = test_df.sample(frac=args.percentage, random_state=30)  # random_state ensures reproducibility

        train_data = process_posts(train_df_sampled, g2v_vectorizer)
        dev_data = process_posts(dev_df_sampled, g2v_vectorizer)
        test_data = process_posts(test_df_sampled, g2v_vectorizer)
    else:
        train_data = process_posts(train_df, g2v_vectorizer)
        dev_data = process_posts(dev_df, g2v_vectorizer)
        test_data = process_posts(test_df, g2v_vectorizer)
        test_df_sampled = test_df

    if args.percentage == 1:  
        g2v_vectorizer.save_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # sep_token_id = tokenizer.sep_token_id
    # print(sep_token_id)
    # exit()
    if args.model_type == "longformer":
        train_dataset = LongformerResidualDataset(train_data, tokenizer)
        dev_dataset = LongformerResidualDataset(dev_data, tokenizer)
        test_dataset = LongformerResidualDataset(test_data, tokenizer)
    elif args.model_type == 'roberta':
        train_dataset = RobertaResidualDataset(train_data, tokenizer)
        dev_dataset = RobertaResidualDataset(dev_data, tokenizer)
        test_dataset = RobertaResidualDataset(test_data, tokenizer)
    elif args.model_type == 'luar':
        train_dataset = LuarResidualDataset(train_data, tokenizer)
        dev_dataset = LuarResidualDataset(dev_data, tokenizer)
        test_dataset = LuarResidualDataset(test_data, tokenizer)
    
    if args.model_type == 'longformer':
        b_size = 32
        print(f"Using batch size of {b_size}")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)
    elif args.model_type == 'roberta':
        b_size = 64
        print(f"Using batch size of {b_size}")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)
    elif args.model_type == 'luar':
        b_size = 128
        print(f"Using batch size of {b_size}")
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=b_size)
        dev_dataloader = DataLoader(dev_dataset, batch_size=b_size)

    test_dataloader = DataLoader(test_dataset, batch_size=1)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=.0001)

    # Training loop
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 3

    # Define the total number of epochs
    num_epochs = 10
    # Include in paper about accumulation steps
    accumulation_steps = 1 # Adjust this based on your GPU capacity and desired batch size
    print(f"number of accuulation steps: {accumulation_steps}")
    
    pbar = tqdm(total=num_epochs, desc="Overall Training Progress", position=0)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_dataloader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", position=1, leave=True)
        optimizer.zero_grad()

        for i, batch in enumerate(train_dataloader):
            with autocast():
                context = {k: v.to(device) for k, v in batch["context"].items()}
                labels = batch["labels"].to(device)  # Move labels to the device
                outputs = model(context, labels=labels)
                loss = outputs["loss"] / accumulation_steps  # Normalize loss to account for accumulation
            
            # breakpoint()
            # print(outputs['logits'])
            loss.backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()  # Update parameters
                optimizer.zero_grad()  # Reset gradients for the next set of accumulation steps

            total_loss += loss.item() * accumulation_steps  # Undo the normalization for logging
            train_dataloader.set_postfix(loss=(total_loss / (i + 1)), refresh=False)
            # optimizer.step()

            # total_loss += loss.item()
            # train_dataloader.set_postfix(loss=loss.item()/len(batch), refresh=False)

        avg_loss = total_loss/len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")

        model.eval()
        val_loss = 0

        # Inside the validation loop, after model.eval()
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            dev_dataloader = tqdm(dev_dataloader, desc=f"Validation Epoch {epoch+1}/{num_epochs}", position=1 ,leave=True)

            for batch in dev_dataloader:
                with autocast():
                    context = {k: v.to(device) for k, v in batch["context"].items()}
                    labels = batch["labels"].to(device)  # Move labels to the device
                    outputs = model(context, labels=labels)
                    loss = outputs["loss"]
                
                val_loss += loss.item()
                # Store predictions and labels to compute MAE and Pearson r
                predictions = outputs["logits"].squeeze().detach().cpu().numpy()  # Adjust based on your model's output
                labels = labels.detach().cpu().numpy()
                val_labels.extend(labels)
                val_predictions.extend(predictions)

        avg_val_loss = val_loss / len(dev_dataloader)
        print(f"Validation Loss: {avg_val_loss}")

        logging.info(f'Epoch {epoch+1}/{num_epochs} completed.')
        # Save the model if the validation loss is the best we've seen so far.
        if avg_val_loss < best_val_loss:
            print("saving model")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"../model/{args.model_type}_{args.run_id}_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                }, f"../model/{args.model_type}_{args.run_id}_checkpoint.pt")
            early_stopping_counter = 0  # reset counter after improvement
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        pbar.update(1)

    pbar.close()

    # TESTING LOOP TO SEE HOW MUCH FINETUNED MODEL CORRECTS GRAM2VEC:
    gram2vec_cosims = []
    vector_map = pd.read_pickle(f'{args.dataset}_vector_map.pkl')

    for i, row in test_df_sampled.iterrows():
        vector1 = vector_map[row['post_1_id']]
        vector2 = vector_map[row['post_2_id']]
        cosim = cosine_similarity(vector1, vector2)
        gram2vec_cosims.append(cosim[0][0])

    predicted_labels = []

    model.eval()
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test loop", position=1 ,leave=True)

        for batch in test_dataloader:
            # print(batch['labels'])
            with autocast():
                context = {k: v.to(device) for k, v in batch["context"].items()}
                labels = batch["labels"].to(device)  # Move labels to the device
                outputs = model(context, labels=labels)
                # print(outputs)
            # print(outputs["logits"])
            predictions = outputs["logits"].squeeze().detach().cpu().numpy()  # Adjust based on your model's output
            predicted_labels.append(predictions)

    residual_cosims = [gram2vec_cosims[i] + predicted_labels[i] for i in range(len(predicted_labels))]
    
    ### ANALYSIS CODE ###
    test_data_for_csv = []
    # Collect data for CSV
    for i in range(len(predicted_labels)):
        test_data_for_csv.append([gram2vec_cosims[i], predicted_labels[i], residual_cosims[i]])

    # Define the CSV file path
    csv_file_path = f"results/analysis_{args.model_type}_{args.run_id}.csv"

    # Define the header for the CSV file
    csv_header = ["gram2vec cosim", "predicted residual", "corrected cosim"]

    # Write the data to the CSV file
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)
        for row in test_data_for_csv:
            writer.writerow(row)  # Add None for the threshold column

    print(f"Test results for analysis saved to {csv_file_path}")
    ### END ANALYSIS CODE ###

    ### EVALAUTION CODE ###
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
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
        stats_line = f"{threshold}, {g2v_correct}/{total}, {long_correct}/{total}, {g2v_correct/len(true_labels):.4f}, {long_correct/len(true_labels):.4f}, "
        stats_line += f"{gram2vec_precision_per_class[0]:.4f}, {gram2vec_precision_per_class[1]:.4f}, {gram2vec_recall_per_class[0]:.4f}, {gram2vec_recall_per_class[1]:.4f}, {gram2vec_f1_per_class[0]:.4f}, {gram2vec_f1_per_class[1]:.4f}, "
        stats_line += f"{precision_per_class[0]:.4f}, {precision_per_class[1]:.4f}, {recall_per_class[0]:.4f}, {recall_per_class[1]:.4f}, {f1_per_class[0]:.4f}, {f1_per_class[1]:.4f}"
        print(stats_line)

        # Define the CSV file path
        csv_file_path = f"results/results_{args.model_type}_{args.run_id}.csv"

        # Define the header for the CSV file
        csv_header = [
            "threshold", "g2v_correct/total", "long_correct/total", "g2v_accuracy", "long_accuracy",
            "gram2vec_diff_author_precision", "gram2vec_same_author_precision",
            "gram2vec_diff_author_recall", "gram2vec_same_author_recall",
            "gram2vec_diff_author_f1", "gram2vec_same_author_f1",
            "residual_diff_author_precision", "residual_same_author_precision",
            "residual_diff_author_recall", "residual_same_author_recall",
            "residual_diff_author_f1", "residual_same_author_f1"
        ]

        # Check if the CSV file already exists
        file_exists = os.path.isfile(csv_file_path)

        # Open the CSV file in append mode
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the header only if the file does not exist
            if not file_exists:
                writer.writerow(csv_header)

            # Write the stats line to the CSV file
            writer.writerow([
                threshold, f"{g2v_correct}/{total}", f"{long_correct}/{total}",
                f"{g2v_correct/len(true_labels):.4f}", f"{long_correct/len(true_labels):.4f}",
                f"{gram2vec_precision_per_class[0]:.4f}", f"{gram2vec_precision_per_class[1]:.4f}",
                f"{gram2vec_recall_per_class[0]:.4f}", f"{gram2vec_recall_per_class[1]:.4f}",
                f"{gram2vec_f1_per_class[0]:.4f}", f"{gram2vec_f1_per_class[1]:.4f}",
                f"{precision_per_class[0]:.4f}", f"{precision_per_class[1]:.4f}",
                f"{recall_per_class[0]:.4f}", f"{recall_per_class[1]:.4f}",
                f"{f1_per_class[0]:.4f}", f"{f1_per_class[1]:.4f}"
            ])
    ### END EVALAUTION CODE ###

    # Plotting the residuals
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_labels, bins=50, color='skyblue', edgecolor='black')
    plt.title('predicted residuals in test')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    plt.savefig(f"predicted_labels_{args.model_type}_{args.run_id}.png")