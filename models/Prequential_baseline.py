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
    df.append(pd.read_pickle(os.path.join(data_dir, f"train_w_rel.pkl")))
    df.append(pd.read_pickle(os.path.join(data_dir, f"valid_w_rel.pkl")))
    df.append(pd.read_pickle(os.path.join(data_dir, f"test_w_rel.pkl")))
    
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


class ClaimData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.evidence = dataframe.evidence
        self.targets = self.data.rating
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        evidence = str(self.evidence[index])
        evidence = " ".join(evidence.split())

        inputs = self.tokenizer.encode_plus(
            text,
            evidence,
            truncation="longest_first",
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.long)
        }

def build_batch(df, batch_indices, max_seq_len, tokenizer):
    
    text = []
    evidence = []
    labels = []
    
    for index in batch_indices:
        text.append(df.text[index])
        evidence.append(df.evidence[index])
        labels.append(df.rating[index])
    
    inputs = tokenizer(
        text,
        evidence,
        truncation="longest_first",
        add_special_tokens=True,
        max_length=max_seq_len,
        pad_to_max_length=True,
    )
    
    input_ids = torch.tensor(inputs["input_ids"], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return input_ids, labels

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()


def f1_loss(logits, labels):
    fc_logits = torch.empty((logits.size(0), 2)).cuda()
    ptc_logits = torch.empty((logits.size(0), 2)).cuda()
    tc_logits = torch.empty((logits.size(0), 2)).cuda()

    fc_logits[:,0] = logits[:,1] + logits[:,2]
    fc_logits[:,1] = logits[:,0]
    
    fc_labels = (labels == 0).long().cuda()

    ptc_logits[:,0] = logits[:,0] + logits[:,2]
    ptc_logits[:,1] = logits[:,1]

    ptc_labels = (labels == 1).long().cuda()

    tc_logits[:,0] = logits[:,0] + logits[:,1]
    tc_logits[:,1] = logits[:,2]

    tc_labels = (labels == 1).long().cuda()

    loss = (f1_loss(fc_logits, fc_labels) + f1_loss(ptc_logits, ptc_labels) + f1_loss(tc_logits, tc_labels)) / 3.0
    
    return loss


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
    
    partitions = load_dataframes(args.data_dir)
    
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
        
        train_dataset = torch.arange(len(train_df)).cuda()
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        
        valid_dataset = torch.arange(len(valid_df)).cuda()
        valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size)
        
        test_dataset = torch.arange(len(test_df)).cuda()
        test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size)
        
        patience_counter = 0
        best_results = None
        epoch = 1
        while patience_counter < args.patience:
        
            train_correct = 0
            train_total = 0
            model.train()
            with torch.set_grad_enabled(True):
                for index, batch in enumerate(train_dataloader):
                    batch_indices = batch.detach().cpu().tolist()
            
                    input_ids, labels = build_batch(train_df, batch_indices, MAX_SEQ_LEN, tokenizer)
                    input_ids = input_ids.cuda()
                    labels = labels.unsqueeze(0).cuda()
            
                    model_output = model(input_ids=input_ids, labels=labels)
            
                    correct_count = torch.sum(torch.argmax(model_output.logits, dim=-1) == labels)

                    train_correct += correct_count
                    train_total += len(batch_indices)
                
                    loss = model_output.loss

                    if ((index + 1) % log_every) == 0 or (index + 1) == len(train_dataloader):
                        print(f"Epoch: {epoch} Batch# ({index+1}/{len(train_dataloader)}), Batch loss: {loss.item():.2f} Correct Predictions: {correct_count}")
                    
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
                labels = []
                for index, batch in enumerate(valid_dataloader):
                    
                    batch_indices = batch.detach().cpu().tolist()
            
                    input_ids, batch_labels = build_batch(valid_df, batch_indices, MAX_SEQ_LEN, tokenizer)
                    input_ids = input_ids.cuda()
                    batch_labels = batch_labels.tolist()
            
                    model_output = model(input_ids=input_ids)
                
                    logits.append(model_output.logits)
                    labels.extend(batch_labels)
                
                logits = torch.vstack(logits)
            
                predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                valid_metrics = compute_metrics((predictions, labels)) 
                
                print("Validation Set.")
                print(valid_metrics)
            
                logits = []
                labels = []
                for index, batch in enumerate(test_dataloader):
                    
                    batch_indices = batch.detach().cpu().tolist()
            
                    input_ids, batch_labels = build_batch(test_df, batch_indices, MAX_SEQ_LEN, tokenizer)
                    input_ids = input_ids.cuda()
                    batch_labels = batch_labels.tolist()
            
                    model_output = model(input_ids=input_ids)
                
                    logits.append(model_output.logits)
                    labels.extend(batch_labels)
                
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
