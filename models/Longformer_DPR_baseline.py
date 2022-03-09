import pandas as pd
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerConfig, LongformerTokenizer, LongformerForSequenceClassification, AdamW
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_dataframes(data_dir, df_key):

    df = {}
    file_path = os.path.join(data_dir, f"train_w_{df_key}.pkl")
    print(f"Loading: {file_path}")
    df["train"] = pd.read_pickle(file_path)
    file_path = os.path.join(data_dir, f"valid_w_{df_key}.pkl")
    print(f"Loading: {file_path}")
    df["valid"] = pd.read_pickle(file_path)
    file_path = os.path.join(data_dir, f"test_w_{df_key}.pkl")
    print(f"Loading: {file_path}")
    df["test"] = pd.read_pickle(file_path)

    return df


class ClaimData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.evidence = dataframe.evidence_sents
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


class SoftF1LossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
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
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)

    target_names = ['False', 'Partially True / False', 'True']
    output_dict = classification_report(labels, predictions, target_names=target_names, output_dict=True)

    return {'accuracy': accuracy, 'macro f1-score': output_dict['macro avg']['f1-score']}


def main(args):
    
    # Set manual seed
    torch.manual_seed(args.rand_seed)
    
    output_dir = os.path.join(args.output_dir, args.exp_name, f"Seed_{args.rand_seed}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    data_dir = os.path.join(args.data_dir, f"Seed_{args.rand_seed}")
    df = load_dataframes(data_dir, args.df_key)
    
    print(f"Output dir: {output_dir}")
    print(f"Data dir: {data_dir}")
    
    model_checkpoint = 'allenai/longformer-base-4096'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    MAX_SEQ_LEN = 512

    training_set = ClaimData(df['train'], tokenizer, MAX_SEQ_LEN)
    validation_set = ClaimData(df['valid'], tokenizer, MAX_SEQ_LEN)
    testing_set = ClaimData(df['test'], tokenizer, MAX_SEQ_LEN)

    NUM_LABELS = 3
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=NUM_LABELS)
    

    if args.loss_fn == "crossent":
        args = TrainingArguments(
            output_dir="output",
            evaluation_strategy = "epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            num_train_epochs=args.num_epochs,
            weight_decay=0.001,
            load_best_model_at_end=True,
            metric_for_best_model='macro f1-score',
            seed=args.rand_seed
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=training_set,
            eval_dataset=validation_set,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
    else:
        args = TrainingArguments(
            output_dir="output",
            evaluation_strategy = "epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=12,
            per_device_eval_batch_size=12,
            num_train_epochs=args.num_epochs,
            weight_decay=0.001,
            load_best_model_at_end=True,
            metric_for_best_model='macro f1-score'
        )
    
    trainer.train()
        
    eval_results = trainer.evaluate(testing_set)
        
    output_file = os.path.join(output_dir, "eval_results.json")
    with open(output_file, "w") as f_obj:
        json.dump(eval_results, f_obj)
                



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Longformer Baseline Model Training & Evaluation')
    
    parser.add_argument('--exp_name', type=str, required=True, help='Please specify the experiment name')
    parser.add_argument('--rand_seed', type=int, required=True, help='Please specify the seed for random number generator')
    parser.add_argument('--num_epochs', type=int, required=True, help='Please specify the number of epochs')
    parser.add_argument('--grad_accum', type=int, required=True, help='Please specify the number of gradient accumulation steps')
    parser.add_argument('--loss_fn', type=str, required=True, choices=["crossent", "f1"], help='Please specify the loss function')
    parser.add_argument('--output_dir', type=str, required=True, help='Please specify the output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Please specify the data directory')
    parser.add_argument('--df_key', type=str, required=True, help='Please specify the dataframe naming key') 
    
    args = parser.parse_args()
    
    main(args)
