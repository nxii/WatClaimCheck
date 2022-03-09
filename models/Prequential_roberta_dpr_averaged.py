import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, AdamW
import numpy as np
import shutil
from tqdm import tqdm
from configparser import ConfigParser
import os
import json
import argparse
import shutil
import sys
import transformers
from nltk.tokenize import sent_tokenize
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from datetime import datetime
import pytz

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    accuracy = accuracy_score(labels, predictions)

    target_names = ['False', 'Partially True / False', 'True']
    output_dict = classification_report(labels, predictions, target_names=target_names, output_dict=True)

    return {'accuracy': accuracy, 'macro f1-score': output_dict['macro avg']['f1-score']}

def generate_dates():
    
    start_date = datetime(2010, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2021, 7, 1, tzinfo=pytz.UTC)
    offset = pd.offsets.DateOffset(months=6)
    dates = [start_date]
    while True:
        next_date = (start_date + offset).to_pydatetime()
        if next_date > end_date:
            return dates
        else:
            dates.append(next_date)
            start_date = next_date
            

def load_dataframes(data_dir):

    df = []
    df.append(pd.read_pickle(os.path.join(data_dir, f"train_w_dpr_ranked_rel.pkl")))
    df.append(pd.read_pickle(os.path.join(data_dir, f"valid_w_dpr_ranked_rel.pkl")))
    df.append(pd.read_pickle(os.path.join(data_dir, f"test_w_dpr_ranked_rel.pkl")))
    
    df = pd.concat(df).reset_index(drop=True)
    
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    dates = generate_dates()
    
    prev_date = None
    partitions = []
    for curr_date in dates:
        if prev_date is None:
            t_df = df.loc[df['date'] < curr_date]
            print(f"< {curr_date}: {len(t_df)}")
        else:
            t_df = df.loc[(df['date'] >= prev_date) & (df['date'] < curr_date)]
            print(f">= {prev_date} and < {curr_date}: {len(t_df)}")
        
        partitions.append(t_df.reset_index(drop=True))
        
        prev_date = curr_date

    return partitions

def load_data(dataframes):

    data = {}    

    for partition in dataframes.keys():
        data[partition] = []

        for _, row in dataframes[partition].iterrows():
            
            scores = row["dpr_scores"]
            
            if partition == "train":
                for article_index in range(len(row["evidence_sents"][:5])):
                    item = {}
                    article_sents = row["evidence_sents"][article_index]
                    article_scores = scores[article_index]
                    
                    assert len(article_sents) == len(article_scores), f"Error! Count mistmatch. Article sents: {len(article_sents)}, article scores: {len(article_scores)}"
                    
                    sorted_indices = np.argsort(article_scores)[::-1].tolist()[:100]
                    
                    sorted_sents = [article_sents[sorted_index] for sorted_index in sorted_indices]
                    
                    item["text"] = row["text"]
                    item["rating"] = row["rating"]
                    item["evidence_sents"] = " ".join(sorted_sents)
                    
                    data[partition].append(item)
            else:
                item_list = []
                for article_index in range(len(row["evidence_sents"][:5])):
                    item = {}
                    article_sents = row["evidence_sents"][article_index]
                    article_scores = scores[article_index]
                    
                    assert len(article_sents) == len(article_scores), f"Error! Count mistmatch. Article sents: {len(article_sents)}, article scores: {len(article_scores)}"
                    
                    sorted_indices = np.argsort(article_scores)[::-1].tolist()[:100]
                    
                    sorted_sents = [article_sents[sorted_index] for sorted_index in sorted_indices]
                    
                    item["text"] = row["text"]
                    item["evidence_sents"] = " ".join(sorted_sents)
                    
                    item_list.append(item)
                
                data[partition].append({
                    "item_list": item_list,
                    "rating": row["rating"]
                })
    return data


def build_tensors(data, indices, max_seq_len, tokenizer):
    
    claim_texts = []
    article_texts = []
    labels = []
    
    for index in indices:
        item = data[index]
        claim_texts.append(item["text"])
        article_texts.append(item["evidence_sents"])
        labels.append(item["rating"])
    
    input_ids = tokenizer(claim_texts, text_pair=article_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt").input_ids
    labels = torch.tensor(labels)
    
    return input_ids, labels


def build_eval_tensors(item_list, max_seq_len, tokenizer):
    
    claim_texts = []
    article_texts = []
    
    for item in item_list:
        claim_texts.append(item["text"])
        article_texts.append(item["evidence_sents"])
    
    input_ids = tokenizer(claim_texts, text_pair=article_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt").input_ids
    
    return input_ids


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


def load_saved(saved_file):
    if not os.path.isfile(saved_file):
        return None
    
    return torch.load(saved_file)


def main(args):
    
    # Set manual seed
    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    
    output_dir = os.path.join(args.output_dir, args.exp_name, f"Seed_{args.rand_seed}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    best_model_file_path = os.path.join(output_dir, "best_model.tar")
    
    data_dir = os.path.join(args.data_dir, f"Seed_{args.rand_seed}")
    
    partitions = load_dataframes(data_dir)
    
    MAX_SEQ_LEN = 512
    
    model_config = RobertaConfig.from_pretrained('roberta-base')
    model_config.num_labels = 3
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=model_config).cuda()
    optimizer = get_optimizer(model)
    
    eval_results = []
    start_period = 0
    
    checkpoint_file_path = os.path.join(args.checkpoint_dir, "checkpoint.tar")
    temp_checkpoint_file_path = os.path.join(args.checkpoint_dir, "temp_checkpoint.tar")
    
    checkpoint = load_saved(checkpoint_file_path)
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        eval_results = checkpoint["eval_results"]
        start_period = len(eval_results)
        
        print(f"Resuming prequential model training from period: {start_period}")
    
    log_every = 100
    for curr_period in range(start_period, len(partitions)-1):
        print()
        print(f"Period: {curr_period}")
        
        train_df, valid_df = train_test_split(partitions[curr_period], test_size=0.2, random_state=args.rand_seed)
        
        if curr_period > 0:
            df_list = partitions[:curr_period] + [train_df]
            train_df = pd.concat(df_list).reset_index(drop=True)
        
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
        test_df = partitions[curr_period+1]
        
        dataframes = {
            "train": train_df,
            "valid": valid_df,
            "test": test_df
        }
        
        data = load_data(dataframes)
        
        index_tensor = torch.arange(len(data["train"]))
        train_dataset = TensorDataset(index_tensor)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

        patience_counter = 0
        best_results = None
        epoch = 1
        while patience_counter < args.patience:
        
            train_correct = 0
            train_total = 0
            model.train()
            with torch.set_grad_enabled(True):
                for index, batch in enumerate(train_dataloader):
                    item_indices = batch[0].tolist()
                    inputs, labels = build_tensors(data["train"], item_indices, MAX_SEQ_LEN, tokenizer)
                    inputs = inputs.cuda()
                    labels = labels.unsqueeze(0).cuda()
                    
                    model_output = model(input_ids=inputs, labels=labels)

                    correct_count = torch.sum(torch.argmax(model_output.logits, dim=-1) == labels)

                    train_correct += correct_count
                    train_total += len(item_indices)
                    
                    loss = model_output.loss

                    if ((index + 1) % log_every) == 0:
                        print(f"Epoch: {epoch} Batch# ({index+1}/{len(train_dataloader)}), Batch loss: {loss.item():.2f} Correct Predictions: {correct_count}")
                        

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                print()
                print(f"Epoch # {epoch+1} Correct Predictions: {(train_correct/train_total)*100:.1f}%")
                print()
            
            print()
            
            print("Validating model.")
            model.eval()
            with torch.no_grad():
            
                logits = []
                labels = []
                for index in tqdm(range(len(data["valid"]))):
                    label = data["valid"][index]["rating"]
                    input_ids = build_eval_tensors(data["valid"][index]["item_list"], MAX_SEQ_LEN, tokenizer)
                    input_ids = input_ids.cuda()
                    
                    model_output = model(input_ids)
                    logits.append(torch.mean(model_output.logits, dim=0, keepdim=True))
                    labels.append(label)
                    
                labels = torch.tensor(labels)
                logits = torch.vstack(logits)
                
                predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                valid_metrics = compute_metrics((predictions, labels)) 
                print("Validation Set.")
                print(valid_metrics)
                
                logits = []
                labels = []
                for index in tqdm(range(len(data["test"]))):
                    label = data["test"][index]["rating"]
                    input_ids = build_eval_tensors(data["test"][index]["item_list"], MAX_SEQ_LEN, tokenizer)
                    input_ids = input_ids.cuda()
                    
                    model_output = model(input_ids)
                    logits.append(torch.mean(model_output.logits, dim=0, keepdim=True))
                    labels.append(label)
                    
                labels = torch.tensor(labels)
                logits = torch.vstack(logits)
                
                predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                test_metrics = compute_metrics((predictions, labels)) 
                print("Test Set.")
                print(test_metrics)
                
            if best_results is None or best_results["valid"]["macro f1-score"] < valid_metrics["macro f1-score"]:
                print("Found new best model.")
                best_results = {
                    "valid": valid_metrics,
                    "test": test_metrics
                }
                print(f"Saving model to file: {best_model_file_path} ")
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, best_model_file_path)
                
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"Current patience: {patience_counter}")
            
                
            epoch += 1
            
        eval_results.append(best_results)
        
        best_model = load_saved(best_model_file_path)
        
        try:
            torch.save(
                {
                    'model_state_dict': best_model["model_state_dict"],
                    'eval_results': eval_results
                }, 
                temp_checkpoint_file_path
            )
            save_successful = True
        except:
            print(f"Warning! Could not save model for period: {curr_period}!")
            save_successful = False
        
        if save_successful:
            shutil.move(temp_checkpoint_file_path, checkpoint_file_path)
        
        
        print("Loading best model.")
        model.load_state_dict(best_model["model_state_dict"])
        optimizer = get_optimizer(model)
    
        output_file = os.path.join(output_dir, "eval_results.json")
        with open(output_file, "w") as f_obj:
            json.dump(eval_results, f_obj, indent=4)
                



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Roberta Baseline Model Training & Evaluation')
    
    parser.add_argument('--exp_name', type=str, required=True, help='Please specify the experiment name')
    parser.add_argument('--rand_seed', type=int, required=True, help='Please specify the seed for random number generator')
    parser.add_argument('--batch_size', type=int, default=12, help='Please specify the batch size')
    parser.add_argument('--patience', type=int, required=True, help='Please specify the patience')
    parser.add_argument('--output_dir', type=str, required=True, help='Please specify the output directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Please specify the output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Please specify the data directory')
    
    args = parser.parse_args()
    
    main(args)
