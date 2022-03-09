import pandas as pd
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
from glob import glob
import sys
import DPR
from DPR import SentenceSimilarityModel

def load_data(data_dir, data_part):
    
    df = pd.read_pickle(os.path.join(data_dir, f"{data_part}_w_rel_sep_articles.pkl"))

    return df


def load_model(model_file):
    
    return torch.load(model_file).cuda()

def get_evidence_scores(model, claim_text, evidence_sents, tokenizer, claim_msl, sent_msl):
    
  claim_ii = tokenizer(claim_text, padding="max_length", truncation=True, max_length=claim_msl, return_tensors="pt").input_ids.cuda()
  sent_ii = tokenizer(evidence_sents, padding="max_length", truncation=True, max_length=sent_msl, return_tensors="pt").input_ids.cuda()

  sent_dataset = TensorDataset(sent_ii)
  sent_dataloader = DataLoader(sent_dataset, shuffle=False, batch_size=256)

  model.eval()
  with torch.no_grad():
    claim_emb = model.embed_claims(claim_ii).squeeze().cpu().numpy()

    sent_embs = []
    for batch in sent_dataloader:
          sent_batch = batch[0]
          sent_emb = model.embed_claims(sent_batch)
          sent_embs.append(sent_emb)
    sent_embs = torch.cat(sent_embs).cpu().numpy()

  sim_scores = sent_embs.dot(claim_emb).tolist()

  return sim_scores


def main(args):
    
    output_dir = os.path.join(os.path.expanduser(args.output_dir), args.exp_name, f"Seed_{args.seed}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    df = load_data(args.data_dir, args.data_part)
    
    model_files = glob(os.path.join(output_dir, "best_*.pt"))
    
    assert len(model_files) == 1, f"Error! Did not find exactly one best model. Found {len(model_files)} file(s)!"
    
    model_file = model_files[0]
    
    model = load_model(model_file)
    
    part = args.data_part
    
    if args.part_set_size == -1:
        part_set_size = df.shape[0]
    else:
        part_set_size = args.part_set_size
        
    if args.part_set_index == -1:
        start_index = 0
        end_index = df.shape[0] + 10 # Any arbitarily higher number than total count
    else:
        start_index = part_set_size * (args.part_set_index - 1)
        end_index = start_index + part_set_size 
        
    
    
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    claim_msl = 120
    sent_msl = 320

    records = []
    
    checkpoint_dir = args.checkpoint_dir
    
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.json")
    temp_checkpoint_file = os.path.join(checkpoint_dir, "temp_checkpoint.json")
    
    if os.path.isfile(checkpoint_file):
        print("Found checkpoint file. Loading checkpoint data...")
        with open(checkpoint_file) as f_obj:
            checkpoint_data = json.load(f_obj)
            start_index = checkpoint_data["start_index"]
            records = checkpoint_data["records"]
    
    print(start_index)
    print(end_index)
    
    for index, row in tqdm(enumerate(df.iterrows())):

        if index < start_index:
            continue

        cols = row[1]
        item = {
            "claim_id": cols["claim_id"],
            "text": cols["text"],
            "evidence_sents": cols["evidence_sents"],
            "tfidf_scores": cols["tfidf_scores"],
            "rating": cols["rating"],
            "date": cols["date"]    
        }
        
        article_sent_counts = []
        evidence_sents = []
        for sents_list in cols["evidence_sents"]:
            article_sent_counts.append(len(sents_list))
            evidence_sents.extend(sents_list)
        
        scores_list = get_evidence_scores(
            model, cols["text"], evidence_sents, tokenizer, claim_msl, sent_msl
        )
        
        dpr_scores = []
        running_count = 0
        for sent_count in article_sent_counts:
            dpr_scores.append(scores_list[running_count:(running_count+sent_count)])
            running_count += sent_count
        
        item["dpr_scores"] = dpr_scores
                
        records.append(item)
        
        if (index + 1) == end_index:
            break
        else:
            if (index + 1) % 100 == 0:
                try:
                    checkpoint_data = {
                        "start_index": index + 1,
                        "records": records
                    }
                    
                    with open(temp_checkpoint_file, "w") as f_obj:
                        json.dump(checkpoint_data, f_obj)
                    
                    save_success = True
                except Exception as e:
                    print("Warning! Could not save the checkpoint data!")
                    print(e)
                    save_success = False
                
                if save_success:
                    shutil.move(temp_checkpoint_file, checkpoint_file)
                    


    new_df = pd.DataFrame.from_records(records)
    
    if args.part_set_index == -1:
        pickle_filename = f"{part}_w_dpr_ranked_rel.pkl"
    else:
        pickle_filename = f"{part}_w_dpr_ranked_rel_{args.part_set_index}.pkl"
    
    pickle_file_path = os.path.join(output_dir, pickle_filename)
    print(f"Saving dataframe pickle to: {pickle_file_path}")
    new_df.to_pickle(pickle_file_path)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Dense Passage Retrieval Model Inference')
    
    parser.add_argument('--exp_name', type=str, required=True, help='Please specify the experiment name')
    parser.add_argument('--seed', type=int, required=True, help='Please specify the seed')
    parser.add_argument('--batch_size', type=int, required=True, help='Please specify the batch size')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Please specify the checkpoint directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Please specify the output directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Please specify the data directory')
    parser.add_argument('--data_part', type=str, required=True, choices=["train", "valid", "test"], help='Please specify the data partition')
    parser.add_argument('--part_set_size', type=int, required=True, help='Please specify the data partition set size')
    parser.add_argument('--part_set_index', type=int, required=True, help='Please specify the data partition set index')
    
    
    args = parser.parse_args()
    
    main(args)
