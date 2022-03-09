#!/usr/bin/env python3 

import os.path
import sys
import pandas as pd
from configparser import ConfigParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import json

def build_vecotrizer(all_sentences):
  vectorizer = TfidfVectorizer()

  X = vectorizer.fit_transform(all_sentences)

  return X, vectorizer

def main(config):
    
    dataset_path = config['default']['dataset_path']
    train_file = config['default']['train_file_w_review_sents']
    
    
    df = pd.read_pickle(os.path.join(dataset_path, train_file))
    
    claim_ids = []
    claim_txt = []
    sentences = []
    sent_ids  = []
    for _, row in df.iterrows():
        claim_ids.append(row["id"])
        claim_txt.append(row["text"])
        sentences.extend(row["review_sentences"])
        sent_ids.extend([(row["id"], index) for index in range(len(row["review_sentences"]))])
        
    
    sent_vecs, vectorizer = build_vecotrizer(sentences)
    claim_vecs = vectorizer.transform(claim_txt)
    
    neg_high_tfidf = {}
    for index in tqdm(range(claim_vecs.shape[0])):
        claim_vec = claim_vecs[index].toarray().squeeze(0)
        claim_id = claim_ids[index]
        
        sim_scores = sent_vecs.dot(claim_vec)
        
        sorted_indices = np.argsort(sim_scores)[::-1]
        
        neg_high_tfidf_list = []
        for sorted_index in sorted_indices:
            if len(neg_high_tfidf_list) == 32:
                break
            
            if sent_ids[sorted_index][0] == claim_id:
                continue
            
            neg_high_tfidf_list.append((sorted_index.item(), sent_ids[sorted_index][0], sent_ids[sorted_index][1]))
                 
        neg_high_tfidf[claim_id] =  neg_high_tfidf_list
    
    file_name = "neg_high_tfidf.json"
    file_path = os.path.join(dataset_path, file_name)
    with open(file_path, "w") as f_obj:
        json.dump(neg_high_tfidf, f_obj, indent=4)
    
    

if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.conf')
    
    main(config)
