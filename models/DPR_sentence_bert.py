import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import RobertaTokenizer
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel, AdamW
import numpy as np
import shutil
from tqdm import tqdm
from configparser import ConfigParser
import os
import json
import argparse
import shutil
import sys

def load_data(data_dir):

    df = {}
    df["train"] = pd.read_pickle(os.path.join(data_dir, "train_w_review_sents.pkl"))
    df["valid"] = pd.read_pickle(os.path.join(data_dir, "valid_w_review_sents.pkl"))
    df["test"] = pd.read_pickle(os.path.join(data_dir, "test_w_review_sents.pkl"))

    claim_ids = {}
    claim_txt = {}
    sentences = {}
    sent_ids = {}
    id2sents = {}
    id2claims = {}

    for partition in df.keys():
        claim_ids[partition] = []
        claim_txt[partition] = []
        sentences[partition] = []
        sent_ids[partition] = []
        id2sents[partition] = {}
        id2claims[partition] = {}

        for _, row in df[partition].iterrows():
            claim_ids[partition].append(row["id"])
            claim_txt[partition].append(row["text"])
            sentences[partition].extend(row["review_sentences"])
            sent_ids[partition].extend([row["id"]] * len(row["review_sentences"]))
            id2sents[partition][row["id"]] = row["review_sentences"]
            id2claims[partition][row["id"]] = row["text"]

    return claim_ids, claim_txt, sentences , sent_ids, id2claims, id2sents


def build_validation_data_loader(
    claim_ids, claim_txt, sentences , sent_ids, claim_msl, sent_msl, tokenizer
):

  partitions = ["valid", "test"]

  ct_input_ids = {}
  ct_attention_mask = {}
  ct_dataset = {}
  ct_dataloader = {}
  for partition in partitions:
    tokenizer_output = tokenizer(claim_txt[partition], padding="max_length", truncation=True, max_length=claim_msl, return_tensors="pt")
    ct_input_ids[partition] = tokenizer_output["input_ids"]
    ct_attention_mask[partition] = tokenizer_output["attention_mask"]
    ct_dataset[partition] = TensorDataset(ct_input_ids[partition], ct_attention_mask[partition])
    ct_dataloader[partition] = DataLoader(ct_dataset[partition], shuffle=False, batch_size=360)

  sent_input_ids = {}
  sent_attention_mask = {}
  sent_dataset = {}
  sent_dataloader = {}
  for partition in partitions:
    tokenizer_output = tokenizer(sentences[partition], padding="max_length", truncation=True, max_length=sent_msl, return_tensors="pt")
    sent_input_ids[partition] = tokenizer_output["input_ids"]
    sent_attention_mask[partition] = tokenizer_output["attention_mask"]
    sent_dataset[partition] = TensorDataset(sent_input_ids[partition], sent_attention_mask[partition])
    sent_dataloader[partition] = DataLoader(sent_dataset[partition], shuffle=False, batch_size=360)

  return ct_dataloader, sent_dataloader


def compute_recall(claim_vecs, sentence_vecs, claim_labels, sentence_labels, top_k_list):

    sentence_labels = np.array(sentence_labels)
    recalls = {top_k: list()  for top_k in top_k_list}
    for index in range(claim_vecs.shape[0]):
        claim_vec = claim_vecs[index]
        claim_label = claim_labels[index]

        sim_scores = sentence_vecs.dot(claim_vec)

        label_sent_indices = np.where(sentence_labels==claim_label)[0]

        sorted_indices = np.argsort(sim_scores)[::-1]

        total_relevant = len(label_sent_indices)
        for top_k in top_k_list:
            retrieved_relevant = len(np.intersect1d(sorted_indices[:top_k], label_sent_indices, assume_unique=True))
            recall = retrieved_relevant / total_relevant

            recalls[top_k].append(recall)

    return recalls

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def main(args):
    
    output_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    claim_ids, claim_txt, sentences , sent_ids, id2claims, id2sents = load_data(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").cuda()

    claim_msl = 120
    sent_msl = 320

    eval_ct_dataloader, eval_sent_dataloader = build_validation_data_loader(
        claim_ids, claim_txt, sentences , sent_ids, claim_msl, sent_msl, tokenizer
    )

    print("Validating model.")
    model.eval()
    with torch.no_grad():
        ct_vecs = {}
        for partition in eval_ct_dataloader.keys():
            ct_vecs[partition] = []
            for batch in tqdm(eval_ct_dataloader[partition]):
                ct_batch = batch[0].cuda()
                ct_att_mask = batch[1].cuda()
                model_output = model(ct_batch)
                embeddings = mean_pooling(model_output, ct_att_mask)
                ct_vecs[partition].append(embeddings.cpu())

            ct_vecs[partition] = torch.cat(ct_vecs[partition]).numpy()

        sent_vecs = {}
        for partition in eval_sent_dataloader.keys():
            sent_vecs[partition] = []
            for batch in tqdm(eval_sent_dataloader[partition]):
                sent_batch = batch[0].cuda()
                sent_att_mask = batch[1].cuda()
                model_output = model(sent_batch)
                embeddings = mean_pooling(model_output, sent_att_mask)
                sent_vecs[partition].append(embeddings.cpu())

            sent_vecs[partition] = torch.cat(sent_vecs[partition]).numpy()

    top_k_list = [10, 25, 50, 100]
    eval_results = {}
    for partition in eval_ct_dataloader.keys():
        print(partition)
        recalls = compute_recall(ct_vecs[partition], sent_vecs[partition], claim_ids[partition], sent_ids[partition], top_k_list)
        eval_results[partition] = {}
        for top_k, recall in recalls.items():
            eval_results[partition][top_k] = np.mean(recall)
            print(f"Top-{top_k} Recall: {round(np.mean(recall),2)*100}")
                    
    print()
    eval_results_filename = f"eval_results.json"
    eval_results_file_path = os.path.join(output_dir, eval_results_filename)
    print(f"Saving eval results to {eval_results_file_path}")
    with open(eval_results_file_path, "w") as f_obj:
        json.dump(eval_results, f_obj, indent=4)
                


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Dense Passage Retrieval Model Training & Evaluation')
    
    parser.add_argument('--exp_name', type=str, required=True, help='Please specify the experiment name')
    parser.add_argument('--output_dir', type=str, required=True, help='Please specify the output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Please specify the data directory')
    
    args = parser.parse_args()
    
    main(args)
