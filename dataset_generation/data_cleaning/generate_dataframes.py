#!/usr/bin/env python3

import numpy as np
import pandas as pd
import json
import os
import sys
import subprocess
from configparser import ConfigParser
from tqdm import tqdm
from nltk import sent_tokenize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def retrieve_cosine_similarity(X, Y):
  #print(X.shape)
  #print(Y.shape)
  #print(cosine_similarity(X, Y).shape)
  sim_scores = cosine_similarity(X, Y).squeeze(0)
  
  return sim_scores.tolist()

def load_articles(dataset_path, articles_path):

  with open(os.path.join(dataset_path, 'train.json')) as f_obj:
    train_records = json.load(f_obj)

  with open(os.path.join(dataset_path, 'valid.json')) as f_obj:
    valid_records = json.load(f_obj)
  
  with open(os.path.join(dataset_path, 'test.json')) as f_obj:
    test_records = json.load(f_obj)
  
  all_records = train_records + valid_records + test_records

  relevant_article_paths = []
  review_article_paths = []
  for record in all_records:
    review_article_json_path = record["label"]["review_article_json_path"]
    assert os.path.isfile(review_article_json_path) == True, f"Error! Review article json file does not exists: {review_article_json_path}"
    review_article_paths.append(review_article_json_path)
    for item in record['metadata']['relevant_articles']:
      for _, article_path in item.items():
        if isinstance(article_path, list):
          article_path = article_path[1]
          relevant_article_paths.append(article_path)

  rel_article_path_to_sents = {}
  all_rel_sentences = []
  for rel_article_path in tqdm(relevant_article_paths):
    
    with open(rel_article_path) as f_obj:
      article_sentences = json.load(f_obj)
        
    rel_article_path_to_sents[rel_article_path] = article_sentences
    all_rel_sentences.extend(article_sentences)
  
  rev_article_path_to_sents = {}
  all_rev_sentences = []
  for rev_article_path in tqdm(review_article_paths):
    
    with open(rev_article_path) as f_obj:
      article_sentences = json.load(f_obj)
        
    rev_article_path_to_sents[rev_article_path] = article_sentences
    all_rev_sentences.extend(article_sentences)

  return rel_article_path_to_sents, all_rel_sentences, rev_article_path_to_sents, all_rev_sentences


def load_data_w_rel(dataset_path, article_path_to_sents, tfidf_vectorizer):
  with open(os.path.join(dataset_path, 'train.json')) as f_obj:
    train_records = json.load(f_obj)
  
  train = []
  for record in tqdm(train_records):
    train_item = {}
    
    train_item["claim_id"] = record['metadata']['id']

    if record['metadata']['claimant'] is not None:
      train_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      train_item['text'] = train_item['text'].strip()
    else:
      train_item['text'] = record['metadata']['claim'].strip()

    X = tfidf_vectorizer.transform([train_item['text']])
    evidence_sentences = []
    evidence_scores = []
    for item in record['metadata']['relevant_articles']:
      for _, article_path in item.items():
        if isinstance(article_path, list):
          article_path = article_path[1]
        else:
          continue
      
        if article_path not in article_path_to_sents:
          continue
      
        article_sentences = [sent.strip() for sent in article_path_to_sents[article_path] if sent.strip() != '']
        
        if len(article_sentences) == 0:
          continue
      
        Y = tfidf_vectorizer.transform(article_sentences) 
        
        sim_scores = retrieve_cosine_similarity(X, Y)
        
        assert len(sim_scores) == len(article_sentences), f"Error! Count mismatch between similarity scores ({len(sim_scores)}) and article senctences ({len(article_sentences)})!"
        
        evidence_sentences.append(article_sentences)
        evidence_scores.append(sim_scores)
    
   
    # Skip if no evidence sentences are available
    if len(evidence_sentences) == 0:
        continue
        
    #print(train_item['text'])
    #print('Top 10 evidence sentences')
    #for index in range(10):
    #  print(evidence_sentences[similar_indices[index]])
    
    #ranked_evidence = [evidence_sentences[sim_index] for sim_index in similar_indices]

    train_item['evidence_sents'] = evidence_sentences
    train_item['tfidf_scores'] = evidence_scores

    train_item['rating'] = record['label']['rating']
    
    if record['metadata']['claim_date'] is not None:
        train_item['date'] = record['metadata']['claim_date']
    else:
        train_item['date'] = record['metadata']['review_date']

    train.append(train_item)
  
  train_df = pd.DataFrame.from_records(train)
  print(train_df)

  with open(os.path.join(dataset_path, 'valid.json')) as f_obj:
    valid_records = json.load(f_obj)

  valid = []
  for record in tqdm(valid_records):
    valid_item = {}
    
    valid_item["claim_id"] = record['metadata']['id']

    if record['metadata']['claimant'] is not None:
      valid_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      valid_item['text'] = valid_item['text'].strip()
    else:
      valid_item['text'] = record['metadata']['claim'].strip()

    evidence_sentences = []
    evidence_scores = []
    for item in record['metadata']['relevant_articles']:
      for _, article_path in item.items():
        if isinstance(article_path, list):
          article_path = article_path[1]
        else:
          continue

        if article_path not in article_path_to_sents:
          continue
      
        article_sentences = [sent.strip() for sent in article_path_to_sents[article_path] if sent.strip() != '']
        
        if len(article_sentences) == 0:
          continue
      
        Y = tfidf_vectorizer.transform(article_sentences) 
        
        sim_scores = retrieve_cosine_similarity(X, Y)
        
        assert len(sim_scores) == len(article_sentences), f"Error! Count mismatch between similarity scores ({len(sim_scores)}) and article senctences ({len(article_sentences)})!"
        
        evidence_sentences.append(article_sentences)
        evidence_scores.append(sim_scores)
          
    # Skip if no evidence sentences are available
    if len(evidence_sentences) == 0:
        continue
        
    valid_item['evidence_sents'] = evidence_sentences
    valid_item['tfidf_scores'] = evidence_scores

    valid_item['rating'] = record['label']['rating']
    
    if record['metadata']['claim_date'] is not None:
        valid_item['date'] = record['metadata']['claim_date']
    else:
        valid_item['date'] = record['metadata']['review_date']

    valid.append(valid_item)
  
  valid_df = pd.DataFrame.from_records(valid)
  print(valid_df)

  with open(os.path.join(dataset_path, 'test.json')) as f_obj:
    test_records = json.load(f_obj)

  test = []
  for record in tqdm(test_records):
    test_item = {}
    
    test_item["claim_id"] = record['metadata']['id']

    if record['metadata']['claimant'] is not None:
      test_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      test_item['text'] = test_item['text'].strip()
    else:
      test_item['text'] = record['metadata']['claim'].strip()

    evidence_sentences = []
    evidence_scores = []
    for item in record['metadata']['relevant_articles']:
      for _, article_path in item.items():
        if isinstance(article_path, list):
          article_path = article_path[1]
        else:
          continue

        if article_path not in article_path_to_sents:
          continue
      
        article_sentences = [sent.strip() for sent in article_path_to_sents[article_path] if sent.strip() != '']
        
        if len(article_sentences) == 0:
          continue
      
        Y = tfidf_vectorizer.transform(article_sentences) 
        
        sim_scores = retrieve_cosine_similarity(X, Y)
        
        assert len(sim_scores) == len(article_sentences), f"Error! Count mismatch between similarity scores ({len(sim_scores)}) and article senctences ({len(article_sentences)})!"
        
        evidence_sentences.append(article_sentences)
        evidence_scores.append(sim_scores)
          
    # Skip if no evidence sentences are available
    if len(evidence_sentences) == 0:
        continue
        
    test_item['evidence_sents'] = evidence_sentences
    test_item['tfidf_scores'] = evidence_scores

    test_item['rating'] = record['label']['rating']
    
    if record['metadata']['claim_date'] is not None:
        test_item['date'] = record['metadata']['claim_date']
    else:
        test_item['date'] = record['metadata']['review_date']

    test.append(test_item)
  
  test_df = pd.DataFrame.from_records(test)

  return train_df, valid_df, test_df

def load_data_w_rev(dataset_path, article_path_to_sents, tfidf_vectorizer):
  with open(os.path.join(dataset_path, 'train.json')) as f_obj:
    train_records = json.load(f_obj)
  
  train = []
  for record in tqdm(train_records):
    train_item = {}

    if record['metadata']['claimant'] is not None:
      train_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      train_item['text'] = train_item['text'].strip()
    else:
      train_item['text'] = record['metadata']['claim'].strip()

    article_path = record["label"]["review_article_json_path"]

    evidence_sentences = []
    filtered_sentences = []
    if article_path in article_path_to_sents:
        original_rating = record["label"]["original_rating"].strip().lower() 
        for sentence in article_path_to_sents[article_path]:
            sentence = sentence.strip().lower()
            if original_rating in sentence:
                filtered_sentences.append(sentence)
            else:
                evidence_sentences.append(sentence)
                
    else:
        continue
    
    # Skip if no evidence sentences are available
    if len(evidence_sentences) == 0:
        continue
        
    X = tfidf_vectorizer.transform([train_item['text']])
    Y = tfidf_vectorizer.transform(evidence_sentences)

    similar_indices = retrieve_most_similar_indices(X, Y)

    #print(train_item['text'])
    #print('Top 10 evidence sentences')
    #for index in range(10):
    #  print(evidence_sentences[similar_indices[index]])

    ranked_evidence = [evidence_sentences[sim_index] for sim_index in similar_indices]
    
    train_item['evidence'] = ranked_evidence
    
    train_item['original_rating'] = record["label"]["original_rating"]
    train_item['filtered_sentences'] = filtered_sentences

    train_item['rating'] = record['label']['rating']

    train.append(train_item)
  
  train_df = pd.DataFrame.from_records(train)

  with open(os.path.join(dataset_path, 'valid.json')) as f_obj:
    valid_records = json.load(f_obj)

  valid = []
  for record in tqdm(valid_records):
    valid_item = {}

    if record['metadata']['claimant'] is not None:
      valid_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      valid_item['text'] = valid_item['text'].strip()
    else:
      valid_item['text'] = record['metadata']['claim'].strip()

    article_path = record["label"]["review_article_json_path"]

    evidence_sentences = []
    filtered_sentences = []
    if article_path in article_path_to_sents:
        original_rating = record["label"]["original_rating"].strip().lower() 
        for sentence in article_path_to_sents[article_path]:
            sentence = sentence.strip().lower()
            if original_rating in sentence:
                filtered_sentences.append(sentence)
            else:
                evidence_sentences.append(sentence)
    else:
        continue
          
    # Skip if no evidence sentences are available
    if len(evidence_sentences) == 0:
        continue
        
    X = tfidf_vectorizer.transform([train_item['text']])
    Y = tfidf_vectorizer.transform(evidence_sentences)

    similar_indices = retrieve_most_similar_indices(X, Y)

    ranked_evidence = [evidence_sentences[sim_index] for sim_index in similar_indices]
    
    valid_item['evidence'] = ranked_evidence
    
    valid_item['original_rating'] = record["label"]["original_rating"]
    valid_item['filtered_sentences'] = filtered_sentences

    valid_item['rating'] = record['label']['rating']

    valid.append(valid_item)
  
  valid_df = pd.DataFrame.from_records(valid)


  with open(os.path.join(dataset_path, 'test.json')) as f_obj:
    test_records = json.load(f_obj)

  test = []
  for record in tqdm(test_records):
    test_item = {}

    if record['metadata']['claimant'] is not None:
      test_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      test_item['text'] = test_item['text'].strip()
    else:
      test_item['text'] = record['metadata']['claim'].strip()

    article_path = record["label"]["review_article_json_path"]

    evidence_sentences = []
    filtered_sentences = []
    if article_path in article_path_to_sents:
        original_rating = record["label"]["original_rating"].strip().lower() 
        for sentence in article_path_to_sents[article_path]:
            sentence = sentence.strip().lower()
            if original_rating in sentence:
                filtered_sentences.append(sentence)
            else:
                evidence_sentences.append(sentence)
    else:
        continue
          
    # Skip if no evidence sentences are available
    if len(evidence_sentences) == 0:
        continue
        
    X = tfidf_vectorizer.transform([train_item['text']])
    Y = tfidf_vectorizer.transform(evidence_sentences)

    similar_indices = retrieve_most_similar_indices(X, Y)

    ranked_evidence = [evidence_sentences[sim_index] for sim_index in similar_indices]
    
    test_item['evidence'] = ranked_evidence
    
    test_item['original_rating'] = record["label"]["original_rating"]
    test_item['filtered_sentences'] = filtered_sentences

    test_item['rating'] = record['label']['rating']

    test.append(test_item)
  
  test_df = pd.DataFrame.from_records(test)

  return train_df, valid_df, test_df

def build_vecotrizer(all_sentences):
  vectorizer = TfidfVectorizer()

  X = vectorizer.fit_transform(all_sentences)

  return X, vectorizer


def main(config):
    
    dataset_path = config['default']['dataset_path']
    articles_path = os.path.join(dataset_path, config['default']['dataset_articles'])
    
    
    print('Loading articles...')
    rel_article_path_to_sents, all_rel_sentences, rev_article_path_to_sents, all_rev_sentences = load_articles(dataset_path, articles_path)

    print('Generating tf idf vectorizer...')
    rel_X, rel_tfidf_vectorizer = build_vecotrizer(all_rel_sentences)
    #rev_X, rev_tfidf_vectorizer = build_vecotrizer(all_rev_sentences)

    print('Generating dataframes with relevant articles...')
    train_w_rel, valid_w_rel, test_w_rel = load_data_w_rel(dataset_path, rel_article_path_to_sents, rel_tfidf_vectorizer)

    train_w_rel.to_pickle(os.path.join(dataset_path, 'train_w_rel.pkl'))
    valid_w_rel.to_pickle(os.path.join(dataset_path, 'valid_w_rel.pkl'))
    test_w_rel.to_pickle(os.path.join(dataset_path, 'test_w_rel.pkl'))
    
    #print('Generating dataframes with review articles...')
    #train_w_rev, valid_w_rev, test_w_rev = load_data_w_rev(dataset_path, rev_article_path_to_sents, rev_tfidf_vectorizer)

    #train_w_rev.to_pickle(os.path.join(dataset_path, 'train_w_rev.pkl'))
    #valid_w_rev.to_pickle(os.path.join(dataset_path, 'valid_w_rev.pkl'))
    #test_w_rev.to_pickle(os.path.join(dataset_path, 'test_w_rev.pkl'))
    
    
if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.conf')
    
    main(config)
 
