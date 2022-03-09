#!/usr/bin/env python3

import json
import os
import sys
import shutil
from datetime import date
from copy import deepcopy
from collections import Counter
from configparser import ConfigParser
from tqdm import tqdm
import numpy as np
import tempfile
import subprocess
from nltk import sent_tokenize
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from rating_mappings import rating_mappings

def retrieve_review_urls(metadata_filename, data_source, data_dir):
    
    data_path = os.path.join(data_dir, data_source)
    
    metadata_file_path  = os.path.join(data_path, metadata_filename)
    with open(metadata_file_path) as f_obj:
        claims_metadata = json.load(f_obj)
        
    review_urls = []
    for claim in claims_metadata:
        review_urls.append(claim['review_url'])
    
    return review_urls
    

def retrieve_article_files(articles_path, claim_id, article_urls, review_urls):
    
    articles_url_to_path = {}
    for article_index, article_url in enumerate(article_urls):
        
        if article_url in review_urls:
            continue
        
        try:
            parsed_url = urlparse(article_url)
        except:
            continue
        
        if parsed_url.netloc.strip() == "":
            continue
        
        if "nytimes.com" in parsed_url.netloc or "washingtonpost.com" in parsed_url.netloc:
            continue
        
        article_file_path = os.path.join(articles_path, f"relevant_{claim_id}_{article_index+1}.html")
        
        if os.path.isfile(article_file_path):
            articles_url_to_path[article_url] = article_file_path
        
    return articles_url_to_path

def retrieve_claims(metadata_filename, data_source, data_dir, articles_dir, rating_mappings, min_articles_req, review_urls):
    
    data_path = os.path.join(data_dir, data_source)
    articles_path = os.path.join(data_path, articles_dir)
    
    metadata_file_path  = os.path.join(data_path, metadata_filename)
    with open(metadata_file_path) as f_obj:
        claims_metadata = json.load(f_obj)
    
    filtered_claims = []
    date_to_claims = {}
    for claim_metadata in claims_metadata:
        
        claim_rating = claim_metadata['rating'].strip().lower()
        # Skip claim if we don't have a mapping for the current claim's rating
        if not claim_rating in rating_mappings:
            continue
        
        # Claim does not have any relevant articles, skip...
        if 'relevant_urls' not in claim_metadata:
            continue
        
        review_article_filename = f"review_{claim_metadata['id']}.html"
        review_article_path = os.path.join(articles_path, review_article_filename)
        
        assert os.path.isfile(review_article_path) == True, "Error! Could not find the review article file! Review article file path: {review_article_path}"
        
        articles_url_to_path = retrieve_article_files(articles_path, claim_metadata['id'], claim_metadata['relevant_urls'], review_urls)
        
        # Check if the no. of relevant articles is above the min threshold
        if len(articles_url_to_path) < min_articles_req: 
            continue
        
        metadata = {}
        
        metadata['claimant'] = claim_metadata['claimant']
        metadata['claim'] = claim_metadata['claim']
        metadata['claim_date'] = claim_metadata['claim_date']
        metadata['review_date'] = claim_metadata['review_date']
        metadata['relevant_articles'] = articles_url_to_path
        
        label = {}
        label['reviewer_name'] = claim_metadata['reviewer_name']
        label['reviewer_site'] = claim_metadata['reviewer_site']
        label['review_url'] = claim_metadata['review_url']
        label['review_article_path'] = review_article_path
        label['rating'] = rating_mappings[claim_rating]
        label['original_rating'] = claim_rating
        
        filtered_claims.append({'metadata': metadata, 'label': label})
    
    return filtered_claims
        
        
        

def main(config):
    
    data_dir = config['default']['data_dir']
    articles_dir = config['default']['articles_dir']
    data_sources = config['default']['data_sources'].split(', ')
    metadata_filename = config['default']['metadata_filename']
    min_articles_req = int(config['default']['min_articles_req'])
    
    dataset_path = config['default']['dataset_path']
    articles_path = os.path.join(dataset_path, config['default']['dataset_articles'])
    
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)
    
    if not os.path.isdir(articles_path):
        os.makedirs(articles_path)
        
    review_urls = []
    for data_source in data_sources:
        review_urls.extend(retrieve_review_urls(metadata_filename, data_source, data_dir))
    
    review_urls = dict(zip(review_urls, [True] * len(review_urls)))
    
    
    all_claims = []
    for data_source in data_sources:
        all_claims.extend(retrieve_claims(metadata_filename, data_source, data_dir, articles_dir, rating_mappings, min_articles_req, review_urls))
        
    
    review_article_urls = {}
    for claim in all_claims:
        review_article_urls[claim['label']['review_url']] = True
        
    #print(len(all_claims))
    #print(all_claims[0].keys())
    #print(json.dumps(all_claims[0], indent=4))
    
    #ratings = [claim['label']['rating'] for claim in all_claims]
    #rating_counter = Counter(ratings)
    #print(rating_counter)
    
    indices = np.arange(len(all_claims))
    
    # randomly shuffle indices
    np.random.shuffle(indices)
    
    dataset = []
    id_to_claim = {}
    for id_, index in enumerate(tqdm(indices)):
        claim = all_claims[index]
        claim['metadata']['id'] = id_ + 1
        claim['label']['id'] = id_ + 1
        
        review_article_html = f"{claim['metadata']['id']}.html"
        review_article_text = f"{claim['metadata']['id']}.txt"
        review_article_json = f"{claim['metadata']['id']}.json"
        
        review_article_html_path = os.path.join(articles_path, review_article_html)
        review_article_text_path = os.path.join(articles_path, review_article_text)
        review_article_json_path = os.path.join(articles_path, review_article_json)
        
        copied_file_path = shutil.copy(claim['label']['review_article_path'], review_article_html_path)
        assert copied_file_path == review_article_html_path, "Error! Could not copy review article to articles folder! src review article path: {claim['label']['review_article_path']} copied file path: {copied_file_path}"
        
        claim['label']['review_article_html_path'] = review_article_html_path
        
        with subprocess.Popen(['links', '-dump', review_article_html_path], stdout=subprocess.PIPE, stderr=open('/dev/null','w'))as proc:
            article_text = proc.stdout.read()
        
        try:
            article_text = article_text.decode('utf-8')
        except:
            article_text = article_text.decode('latin-1')
                
            
        with open(review_article_text_path, "w") as f_obj:
            f_obj.write(article_text)
            
        try:
            article_sentences = sent_tokenize(article_text)
            article_sentences = [sent.strip() for sent in article_sentences if sent.strip() != '']
        
            with open(review_article_json_path, 'w') as f_obj:
                json.dump(article_sentences, f_obj)
        except:
            pass
        
        claim['label']['review_article_json_path'] = review_article_json_path
        
        relevant_articles = []
        for article_index, (article_url, article_path) in enumerate(claim['metadata']['relevant_articles'].items()):
            relevant_article_html = f"{claim['metadata']['id']}_{article_index+1}.html"
            relevant_article_text = f"{claim['metadata']['id']}_{article_index+1}.txt"
            relevant_article_json = f"{claim['metadata']['id']}_{article_index+1}.json"
            
            relevant_article_html_path = os.path.join(articles_path, relevant_article_html)
            relevant_article_text_path = os.path.join(articles_path, relevant_article_text)
            relevant_article_json_path = os.path.join(articles_path, relevant_article_json)

            copied_file_path = shutil.copy(article_path, relevant_article_html_path)
            
            assert copied_file_path == relevant_article_html_path, "Error! Could not copy relevant article to articles folder! src relevant article path: {article} copied file path: {copied_file_path}"
            
            
            with subprocess.Popen(['links', '-dump', relevant_article_html_path], stdout=subprocess.PIPE, stderr=open('/dev/null','w')) as proc:
                article_text = proc.stdout.read()
                
            try:
                article_text = article_text.decode('utf-8')
            except:
                article_text = article_text.decode('latin-1')
                
            with open(relevant_article_text_path, "w") as f_obj:
                f_obj.write(article_text)
            
            try:
                article_sentences = article_text.split('\n')
                article_sentences = [sent.strip() for sent in article_sentences if sent.strip() != '']
            
                with open(relevant_article_json_path, 'w') as f_obj:
                    json.dump(article_sentences, f_obj)
            except:
                pass
                
            relevant_articles.append({article_url: [relevant_article_html_path, relevant_article_json_path]})
            
            relevant_articles.append({article_url: relevant_article_html_path})
        
        claim['metadata']['relevant_articles'] = relevant_articles
        
        
        '''
        if claim['metadata']['claim_date'] is not None:
            claim_date = date.fromisoformat(claim['metadata']['claim_date'])
        else:
            claim_date = date.fromisoformat(claim['metadata']['review_date'])
        
        if claim_date in date_to_claim_ids:
            date_to_claim_ids[claim_date].append(claim['metadata']['id'])
        else:
            date_to_claim_ids[claim_date] = [claim['metadata']['id']]
        '''
        
        id_to_claim[claim['metadata']['id']] = claim
        
        dataset.append(claim)
    
    train_dataset = dataset[:int(len(dataset)*0.8)]
    dev_dataset = dataset[int(len(dataset)*0.8):]
    
    valid_dataset = dev_dataset[:int(len(dev_dataset)*0.5)]
    test_dataset = dev_dataset[int(len(dev_dataset)*0.5):]
    
    print(f'Dataset contains: {len(dataset)} claims.')
    print(f'Train set contains: {len(train_dataset)} claims.')
    print(f'Valid set contains: {len(valid_dataset)} claims.')
    print(f'Test set contains: {len(test_dataset)} claims.')
    
    train_file_path = os.path.join(dataset_path, 'train.json')
    with open(train_file_path, 'w') as f_obj:
        json.dump(train_dataset, f_obj, indent=4)
    
    valid_file_path = os.path.join(dataset_path, 'valid.json')
    with open(valid_file_path, 'w') as f_obj:
        json.dump(valid_dataset, f_obj, indent=4)
    
    test_file_path = os.path.join(dataset_path, 'test.json')
    with open(test_file_path, 'w') as f_obj:
        json.dump(test_dataset, f_obj, indent=4)
    
    '''
    d2cid_file_path = os.path.join(dataset_path, 'date_to_claim_ids.json')
    with open(d2cid_file_path, 'w') as f_obj:
        json.dump(date_to_claim_ids, f_obj, indent=4)
    '''
    
    id2c_file_path = os.path.join(dataset_path, 'id_to_claim.json')
    with open(id2c_file_path, 'w') as f_obj:
        json.dump(id_to_claim, f_obj, indent=4)
        
        

if __name__ == '__main__':
    
    config = ConfigParser()
    config.read('config.conf')
    
    main(config)
