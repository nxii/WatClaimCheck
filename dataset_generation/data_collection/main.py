#!/usr/bin/env python3

import argparse
import json
from configparser import ConfigParser
from data_sources import *

def main(args, config):
    
    google_fc_sources = {
        'AfricaCheck': 'africacheck.org', 
        'FactCheck': 'factcheck.org', 
        'AFP': 'factcheck.afp.com', 
        'BBC': 'bbc.co.uk', 
        'NYTimes': 'nytimes.com', 
        'USAToday': 'usatoday.com', 
        'AltNews': 'altnews.in', 
        'FullFact': 'fullfact.org'
    }
    
    ds_class = globals()[args.data_source + 'DS']
    
    if args.data_source in google_fc_sources:
        ds_obj = ds_class(
            data_source=args.data_source,
            publisher_site=google_fc_sources[args.data_source],
            api_key=config['default']['google_api_key'],
            claim_search_endpoint=config['default']['google_claim_search_endpoint'],
            data_dir=config['default']['data_dir'],
            articles_dir=config['default']['articles_dir'],
            claim_metadata_file=config['default']['claim_metadata_file'],
            claim_metadata_file_w_relevant=config['default']['claim_metadata_file_w_relevant'],
            retries=int(config['default']['retries'])
        )
    else:
        ds_obj = ds_class(
            data_source=args.data_source,
            claim_search_endpoint=config['default'][args.data_source + '_claim_search_endpoint'],
            data_dir=config['default']['data_dir'],
            articles_dir=config['default']['articles_dir'],
            claim_metadata_file=config['default']['claim_metadata_file'],
            claim_metadata_file_w_relevant=config['default']['claim_metadata_file_w_relevant'],
            retries=int(config['default']['retries'])
        )
        
    
    if args.data_type == 'metadata':
        ds_obj.retrieveClaimsMetadata()
    elif args.data_type == 'review':
        ds_obj.loadClaimsMetadata()
        ds_obj.retrieveReviewArticle()
    else:
        ds_obj.loadClaimsMetadata(with_relevant=True)
        ds_obj.retrieveRelevantArticles() 
        
        

if __name__ == '__main__':
    
    config = ConfigParser()
    config.read('config.conf')
    
    data_sources = ['Politifact', 'Snopes', 'AfricaCheck', 'FactCheck', 'AFP', 'BBC', 'USAToday', 'AltNews', 'FullFact']
    
    parser = argparse.ArgumentParser(description='Fact Check Dataset Raw Data Retrieval')
    parser.add_argument('data_source', metavar='DATA_SOURCE', type=str, choices=data_sources, help='Please specify the data source')
    parser.add_argument('data_type', metavar='DATA_TYPE', type=str, choices=['metadata', 'review', 'relevant'], help='Specifies the type of data that is to be retrieved')
    
    args = parser.parse_args()
    
    main(args, config)
