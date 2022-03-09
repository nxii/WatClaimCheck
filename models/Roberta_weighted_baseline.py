import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, AdamW
import numpy as np
import shutil
from tqdm import tqdm
from configparser import ConfigParser
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
import json
import argparse
import shutil


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    accuracy = accuracy_score(labels, predictions)

    target_names = ['False', 'Partially True / False', 'True']
    output_dict = classification_report(labels, predictions, target_names=target_names, output_dict=True)

    return {'accuracy': accuracy, 'macro f1-score': output_dict['macro avg']['f1-score']}


def load_data(data_dir, score_type):

    print("Loading data files...")
    df = {}
    df["train"] = pd.read_pickle(os.path.join(data_dir, "train_w_dpr_ranked_rel.pkl"))
    df["valid"] = pd.read_pickle(os.path.join(data_dir, "valid_w_dpr_ranked_rel.pkl"))
    df["test"] = pd.read_pickle(os.path.join(data_dir, "test_w_dpr_ranked_rel.pkl"))
    
    print("Processing data...")
    data = {}    

    for partition in df.keys():
        data[partition] = []

        for _, row in df[partition].iterrows():
            
            if score_type == "tfidf":
                scores = row["tfidf_scores"]
            else:
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


def build_train_data_loader(
    train_data, max_seq_len, batch_size, tokenizer
):
    
    print("Building train data loader...")
    
    claim_texts = []
    article_texts = []
    labels = []
  
    for item in train_data:
        claim_texts.append(item["text"])
        article_texts.append(item["evidence_sents"])
        labels.append(item["rating"])
        
    input_ids = tokenizer(claim_texts, text_pair=article_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt").input_ids
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    return dataloader

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


def load_model_checkpoint(model_checkpoint_file):
    if not os.path.isfile(model_checkpoint_file):
        return None
    
    return torch.load(model_checkpoint_file)


def main(args):
    
    # Set manual seed
    torch.manual_seed(args.rand_seed)
    
    output_dir = os.path.join(args.output_dir, args.exp_name, f"Seed_{args.rand_seed}")
    print(f"Output dir: {output_dir}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    data_dir = os.path.join(args.data_dir, f"Seed_{args.rand_seed}")
    print(f"Data dir: {data_dir}")
    
    data = load_data(data_dir, args.score_type)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    max_seq_len = 512
    
    index_tensor = torch.arange(len(data["train"]))
    train_dataset = TensorDataset(index_tensor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    
    #train_dataloader = build_train_data_loader(data["train"], max_seq_len, args.batch_size, tokenizer)
    
    model_config = RobertaConfig.from_pretrained('roberta-base')
    model_config.num_labels = 3
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=model_config).cuda()
    optimizer = get_optimizer(model)
    
    model_checkpoint_file = os.path.join(args.checkpoint_dir, "model.tar")
    temp_checkpoint_file = os.path.join(args.checkpoint_dir, "temp_model.tar")
    
    model_checkpoint = load_model_checkpoint(model_checkpoint_file)
    
    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint["model_state_dict"])
        optimizer.load_state_dict(model_checkpoint['optimizer_state_dict'])
        start_epoch = model_checkpoint["epoch"] + 1
        eval_results = model_checkpoint["eval_results"]
        print(f"Resuming model training from epoch: {start_epoch}")
        
        """
        if len(eval_results) != (start_epoch - 1):
            print("Validating model.")
            model.eval()
            with torch.no_grad():
               
               logits = []
               labels = []
               for index in tqdm(range(len(data["valid"]))):
                   label = data["valid"][index]["rating"]
                   input_ids = build_eval_tensors(data["valid"][index]["item_list"], max_seq_len, tokenizer)
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
                   input_ids = build_eval_tensors(data["test"][index]["item_list"], max_seq_len, tokenizer)
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
               
               eval_results[start_epoch] = {"valid": valid_metrics, "test": test_metrics}
        """
    else:
        start_epoch = 1
        eval_results = {}
        
    log_every = 100
    
    
    print("Starting model training.")
    for epoch in range(start_epoch, 11):
        train_correct = 0
        train_total = 0
        model.train()
        with torch.set_grad_enabled(True):
            for index, batch in enumerate(train_dataloader):
                item_indices = batch[0].tolist()
                inputs, labels = build_tensors(data["train"], item_indices, max_seq_len, tokenizer)
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
                input_ids = build_eval_tensors(data["valid"][index]["item_list"], max_seq_len, tokenizer)
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
                input_ids = build_eval_tensors(data["test"][index]["item_list"], max_seq_len, tokenizer)
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
            
            eval_results[epoch] = {"valid": valid_metrics, "test": test_metrics}
            
        try:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'eval_results': eval_results
                }, 
                temp_checkpoint_file
            )
            save_successful = True
        except:
            print(f"Warning! Could not save model for epoch: {epoch+1}!")
            save_successful = False
        
        if save_successful:
            shutil.move(temp_checkpoint_file, model_checkpoint_file)

    print()
    eval_results_filename = f"eval_results.json"
    eval_results_file_path = os.path.join(output_dir, eval_results_filename)
    print(f"Saving eval results to {eval_results_file_path}")
    with open(eval_results_file_path, "w") as f_obj:
        json.dump(eval_results, f_obj, indent=4)
            



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Dense Passage Retrieval Model Training & Evaluation')
    
    parser.add_argument('--exp_name', type=str, required=True, help='Please specify the experiment name')
    parser.add_argument('--rand_seed', type=int, required=True, help='Please specify the seed for random number generator')
    parser.add_argument('--batch_size', type=int, required=True, help='Please specify the batch size')
    parser.add_argument('--score_type', type=str, choices=["tfidf", "dpr"], required=True, help='Please specify the score type to use')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Please specify the checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Please specify the output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Please specify the data directory')
    
    args = parser.parse_args()
    
    main(args)
