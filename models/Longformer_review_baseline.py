import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerConfig, LongformerTokenizer, LongformerForSequenceClassification, AdamW
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


def load_data(data_dir):

    print("Loading data files...")
    df = {}
    df["train"] = pd.read_pickle(os.path.join(data_dir, "train_w_rev.pkl"))
    df["valid"] = pd.read_pickle(os.path.join(data_dir, "valid_w_rev.pkl"))
    df["test"] = pd.read_pickle(os.path.join(data_dir, "test_w_rev.pkl"))
    
    print("Processing data...")
    data = {}    

    for partition in df.keys():
        data[partition] = []

        for _, row in df[partition].iterrows():
            
            item = {}
            
            item["text"] = row["text"]
            item["rating"] = row["rating"]
            item["evidence_sents"] = " ".join(row["filtered_sentences"])
            
            data[partition].append(item)

    return data


def build_dataloaders(
    data, max_seq_len, batch_size, tokenizer
):
    
    print("Building data loaders...")
    
    dataloader = {}
    for split in data.keys():
        claim_texts = []
        article_texts = []
        labels = []
    
        for item in data[split]:
            claim_texts.append(item["text"])
            article_texts.append(item["evidence_sents"])
            labels.append(item["rating"])
            
        input_ids = tokenizer(claim_texts, text_pair=article_texts, padding="max_length", truncation=True, max_length=max_seq_len, return_tensors="pt").input_ids
        labels = torch.tensor(labels)
        dataset = TensorDataset(input_ids, labels)
        
        if split == "train":
            dataloader[split] = DataLoader(dataset, shuffle=True, batch_size=batch_size)
        else:
            dataloader[split] = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    return dataloader

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
    
    data_dir = args.data_dir
    print(f"Data dir: {data_dir}")
    
    data = load_data(data_dir)
    
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    max_seq_len = 400
    
    dataloader = build_dataloaders(data, max_seq_len, args.batch_size, tokenizer)
    
    model_config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
    model_config.num_labels = 3
    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', config=model_config).cuda()
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
            for index, batch in enumerate(dataloader["train"]):
                inputs = batch[0].cuda()
                labels = batch[1].unsqueeze(0).cuda()
                
                model_output = model(input_ids=inputs, labels=labels)

                correct_count = torch.sum(torch.argmax(model_output.logits, dim=-1) == labels)

                train_correct += correct_count
                train_total += labels.size(1)
                
                loss = model_output.loss

                if ((index + 1) % log_every) == 0:
                    print(f"Epoch: {epoch} Batch# ({index+1}/{len(dataloader['train'])}), Batch loss: {loss.item():.2f} Correct Predictions: {correct_count}")
                    

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
            print()
            print(f"Epoch # {epoch} Correct Predictions: {(train_correct/train_total)*100:.1f}%")
            print()
            
        print()
        print("Validating model.")
        model.eval()
        with torch.no_grad():
            
            logits = []
            label_list = []
            for index, batch in tqdm(enumerate(dataloader["valid"])):
                inputs = batch[0].cuda()
                labels = batch[1].unsqueeze(0).cuda()
                
                model_output = model(inputs)
                logits.append(model_output.logits)
                label_list.extend(labels.squeeze(0).detach().cpu().tolist())
                
            logits = torch.vstack(logits)
            
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            valid_metrics = compute_metrics((predictions, label_list)) 
            print("Validation Set.")
            print(valid_metrics)
            
            logits = []
            label_list = []
            for index, batch in tqdm(enumerate(dataloader["test"])):
                inputs = batch[0].cuda()
                labels = batch[1].unsqueeze(0).cuda()
                
                model_output = model(inputs)
                logits.append(model_output.logits)
                label_list.extend(labels.squeeze(0).detach().cpu().tolist())
                
            logits = torch.vstack(logits)
            
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            test_metrics = compute_metrics((predictions, label_list)) 
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
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Please specify the checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Please specify the output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Please specify the data directory')
    
    args = parser.parse_args()
    
    main(args)
