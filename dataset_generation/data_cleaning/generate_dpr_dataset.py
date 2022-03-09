#!/usr/bin/env python3

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import json
import os
import sys
import subprocess
from configparser import ConfigParser
from urllib.parse import urlparse
from collections import Counter
from tqdm import tqdm
from nltk import sent_tokenize


def remove_irrelevant_tags(node):
    script_nodes = node.find_all('script')
    for script_node in script_nodes:
        script_node.decompose()
    
    style_nodes = node.find_all('style')
    for style_node in style_nodes:
        style_node.decompose()
        

def extract_altnews_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_node = soup.select("#main > div.m-hentry-s.m-h-s.post > div > div")[0]
    
    assert article_node is not None, "Error! could not find the article text node!"
    
    t_node = article_node.find("div", id="enhancedtextwidget-12")
    if t_node is not None:
        t_node.decompose()
        
    t_node = article_node.find("div", id="jp-relatedposts")
    if t_node is not None:
        t_node.decompose()
    
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()

def extract_factcheck_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_node = soup.find('div', class_="entry-content")
    
    assert article_node is not None, "Error! could not find the article text node!"
    
    h5_sources = article_node.find("h5", string="Sources")
    if h5_sources is not None:
        prev_sibling_node = h5_sources.find_previous_sibling("p")
        if prev_sibling_node is not None:
            prev_sib_text = prev_sibling_node.get_text()
            if "Editor's note" in prev_sib_text or "This fact check is available at IFCN’s" in prev_sib_text:
                prev_sibling_node.decompose()
                
                prev_sibling_node = h5_sources.find_previous_sibling("p")
                if prev_sibling_node is not None:
                    prev_sib_text = prev_sibling_node.get_text()
                    if "Editor’s Note:" in prev_sib_text or "This fact check is available at IFCN’s" in prev_sib_text:
                        prev_sibling_node.decompose()
            
        source_p_nodes = h5_sources.find_next_siblings("p")
        for source_p_node in source_p_nodes:
            source_p_node.decompose()
    
        h5_sources.decompose()
    
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()

def extract_politifact_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_node = soup.find('article', class_="m-textblock")
    
    assert article_node is not None, "Error! could not find the article text node!"
    
    stf_node = article_node.find("div", class_="artembed")
    if stf_node is not None:
        stf_node.decompose()
    
    ffc_node = article_node.find("section", class_="o-pick")
    if ffc_node is not None:
        ffc_node.decompose()
        
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()

def extract_snopes_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_nodes = soup.select("div.single-body.card.card-body.rich-text")
    
    assert len(article_nodes) > 0, "Error! could not find the article text node!"
    
    article_node = article_nodes[0]
    
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()

def extract_afp_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_node = soup.find('div', class_="article-entry")
    
    assert article_node is not None, "Error! could not find the article text node!"
    
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()

def extract_africacheck_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_nodes = soup.select("#block-mainpagecontent > article > div > div.cell.medium-8.article--main")
    
    assert len(article_nodes) > 0, "Error! could not find the article text node!"
    
    article_node = article_nodes[0]
    
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()

def extract_usatoday_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_node = soup.find("article")
    
    assert article_node is not None, "Error! could not find the article text node!"
    
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()
    
def extract_fullfact_review_article(review_html):
    
    soup = BeautifulSoup(review_html, 'html.parser')
    
    article_node = soup.find("article")
    
    assert article_node is not None, "Error! could not find the article text node!"
    
    remove_irrelevant_tags(article_node)
    
    return article_node.get_text().strip()

def extract_review_text(review_url, review_article_path):
    
    review_text_extractors = {
        "www.altnews.in": extract_altnews_review_article,
        "www.factcheck.org": extract_factcheck_review_article,
        "www.politifact.com": extract_politifact_review_article,
        "www.snopes.com": extract_snopes_review_article,
        "factcheck.afp.com": extract_afp_review_article,
        "africacheck.org": extract_africacheck_review_article,
        "www.usatoday.com": extract_usatoday_review_article,
        "fullfact.org": extract_fullfact_review_article
    }
    
    with open(review_article_path) as f_obj:
        article_html = f_obj.read()
    
    url_parts = urlparse(review_url)
    host = url_parts.netloc
    
    try:
        article_text = review_text_extractors[host](article_html)
    except Exception as e:
        print("Following exception occurred while extracting review article text!")
        print(e)
        print(f"Review article URL: {review_url}")
        print(article_html)
        sys.exit(1)
    
    article_sentences = sent_tokenize(article_text)
    article_sentences = [sent.strip() for sent in article_sentences if sent.strip() != '']
    
    return article_sentences
    
    
def load_articles(dataset_path, articles_path):
    
    with open(os.path.join(dataset_path, 'train.json')) as f_obj:
        train_records = json.load(f_obj)

    with open(os.path.join(dataset_path, 'valid.json')) as f_obj:
        valid_records = json.load(f_obj)
  
    with open(os.path.join(dataset_path, 'test.json')) as f_obj:  
        test_records = json.load(f_obj)
  
    all_records = train_records + valid_records + test_records

    review_articles = []
    for record in all_records:
        info = {
            "url": record["label"]["review_url"],
            "path": record["label"]["review_article_html_path"]
        }
        assert os.path.isfile(info["path"]) == True, f"Error! Review article json file does not exists: {review_article_json_path}"
        review_articles.append(info)
    
    rev_article_path_to_sents = {}
    all_rev_sentences = []
    hosts = []
    for rev_article in tqdm(review_articles):
        article_sentences = extract_review_text(rev_article["url"], rev_article["path"])
        file_path, file_name = os.path.split(rev_article["path"])
        file_pre, file_ext = os.path.splitext(file_name)
                
        sent_file_name = f"{file_pre}_sent.json"
        sent_file_path = os.path.join(file_path, sent_file_name)
        
        with open(sent_file_path, "w") as f_obj:
            json.dump(article_sentences, f_obj, indent=4)
            
        rev_article_path_to_sents[rev_article["path"]] = article_sentences
        
    return rev_article_path_to_sents


def load_data_w_review_sents(dataset_path, article_path_to_sents):
  with open(os.path.join(dataset_path, 'train.json')) as f_obj:
    train_records = json.load(f_obj)
  
  train = []
  for record in tqdm(train_records):
    train_item = {"id": record["metadata"]["id"]}
    
    if record['metadata']['claimant'] is not None:
      train_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      train_item['text'] = train_item['text'].strip()
    else:
      train_item['text'] = record['metadata']['claim'].strip()

    article_path = record["label"]["review_article_html_path"]

    if article_path in article_path_to_sents:
        train_item["review_sentences"] = article_path_to_sents[article_path]
    else:
        continue
    
    train.append(train_item)
  
  train_df = pd.DataFrame.from_records(train)

  with open(os.path.join(dataset_path, 'valid.json')) as f_obj:
    valid_records = json.load(f_obj)

  valid = []
  for record in tqdm(valid_records):
    valid_item = {"id": record["metadata"]["id"]}

    if record['metadata']['claimant'] is not None:
      valid_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      valid_item['text'] = valid_item['text'].strip()
    else:
      valid_item['text'] = record['metadata']['claim'].strip()

    article_path = record["label"]["review_article_html_path"]

    if article_path in article_path_to_sents:
        valid_item["review_sentences"] = article_path_to_sents[article_path]
    else:
        continue
    
    valid.append(valid_item)
  
  valid_df = pd.DataFrame.from_records(valid)


  with open(os.path.join(dataset_path, 'test.json')) as f_obj:
    test_records = json.load(f_obj)

  test = []
  for record in tqdm(test_records):
    test_item = {"id": record["metadata"]["id"]}

    if record['metadata']['claimant'] is not None:
      test_item['text'] = record['metadata']['claim'].strip() + ' ' + record['metadata']['claimant'].strip() 
      test_item['text'] = test_item['text'].strip()
    else:
      test_item['text'] = record['metadata']['claim'].strip()

    article_path = record["label"]["review_article_html_path"]

    if article_path in article_path_to_sents:
        test_item["review_sentences"] = article_path_to_sents[article_path]
    else:
        continue
    
    test.append(test_item)
  
  test_df = pd.DataFrame.from_records(test)

  return train_df, valid_df, test_df


def main(config):
    
    dataset_path = config['default']['dataset_path']
    articles_path = os.path.join(dataset_path, config['default']['dataset_articles'])
    
    
    print('Loading articles...')
    rev_article_path_to_sents = load_articles(dataset_path, articles_path)

    print('Generating dataframes with review article sentences...')
    train_w_rev, valid_w_rev, test_w_rev = load_data_w_review_sents(dataset_path, rev_article_path_to_sents)

    train_w_rev.to_pickle(os.path.join(dataset_path, 'train_w_review_sents.pkl'))
    valid_w_rev.to_pickle(os.path.join(dataset_path, 'valid_w_review_sents.pkl'))
    test_w_rev.to_pickle(os.path.join(dataset_path, 'test_w_review_sents.pkl'))
    
    
if __name__ == "__main__":
    config = ConfigParser()
    config.read('config.conf')
    
    main(config)
 
