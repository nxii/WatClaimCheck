#!/usr/bin/env python3

import datetime
import time
import json
import sys
import os
from tqdm import tqdm
from configparser import ConfigParser

from utils import get_request
from data_source import DataSource


class GoogleFactCheckDS(DataSource):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
        
        self.api_key = api_key
        self.publisher_site = publisher_site
        
    
    def retrieveClaimsMetadata(self):
        
        get_params = {
           'key': self.api_key,
           'pageSize': 1000,
           'languageCode': 'en'           
        }
        
        claims_metadata = []
        
            
        print(f'Retrieving claims for {self.data_source}...')
        
        get_params['reviewPublisherSiteFilter'] = self.publisher_site
        
        all_claims = []
        
        while True:
            r = get_request(self.claim_search_endpoint, self.retries, request_params=get_params)
            assert r is not None, f'Error! API request failed. Request parameters: {get_params}'
            time.sleep(5)
            json_response = r.json()
            
            all_claims.extend(json_response['claims'])
                        
            if 'nextPageToken' in json_response:
                get_params['pageToken'] = json_response['nextPageToken']
            else:
                break
        
        all_claims_metadata = []
        both_missing = 0
        cd_missing = 0
        rd_missing = 0
        for index in range(len(all_claims)):
            claim_metadata = {}
            claim_metadata['claim'] = all_claims[index]['text']
            if 'claimant' in all_claims[index]:
                claim_metadata['claimant'] = all_claims[index]['claimant']
            else:
                claim_metadata['claimant'] = None
                
            if 'claimDate' in all_claims[index]:
                claim_metadata['claim_date'] = datetime.datetime.strptime(all_claims[index]['claimDate'], "%Y-%m-%dT%H:%M:%S%z").date().isoformat()
            else:
                claim_metadata['claim_date'] = None
            
            found_publisher = False
            if len(all_claims[index]['claimReview']) == 1:
                claim_review = all_claims[index]['claimReview'][0]
                found_publisher = True
            else:
                for claim_review in all_claims[index]['claimReview']:
                    if self.publisher_site == 'bbc.co.uk':
                        if claim_review['publisher']['site'].lower().find('bbc.co.uk') != -1 or claim_review['publisher']['site'].lower().find('bbc.com') != -1:
                            found_publisher = True
                            break
                    else:
                        if claim_review['publisher']['site'].lower().find(self.publisher_site) != -1:
                            found_publisher = True
                            break
            
            assert found_publisher == True, f'Error! Could not find the publisher site!\n' + str(all_claims[index]['claimReview'])
            
            claim_metadata['reviewer_name'] = claim_review['publisher']['name']
            claim_metadata['reviewer_site'] = claim_review['publisher']['site']
            
            claim_metadata['review_url'] = claim_review['url']
            claim_metadata['review_title'] = claim_review['title']
            claim_metadata['rating'] = claim_review['textualRating']
            
            if 'reviewDate' in claim_review:
                claim_metadata['review_date'] = datetime.datetime.strptime(claim_review['reviewDate'], "%Y-%m-%dT%H:%M:%S%z").date().isoformat()
            else:
                claim_metadata['review_date'] = None
                
            
            if claim_metadata['claim_date'] is None and claim_metadata['review_date'] is None:
                both_missing += 1
            elif claim_metadata['claim_date'] is None:
                cd_missing += 1
            elif claim_metadata['review_date'] is None:
                rd_missing += 1
            
            all_claims_metadata.append(claim_metadata)
                
         
        self.claims_metadata = all_claims_metadata
            
        total_claims = len(self.claims_metadata)
        
        print(f'Total claims retrieved: {total_claims}')
        
        print('Saving claims...')
        self.saveClaimsMetadata()
    
    
    def retrieveReviewArticle(self):
        
        for index, claim_metadata in enumerate(self.claims_metadata):
            claim_metadata['id'] = index + 1
        
        ds_dir = os.path.join(self.data_dir, self.data_source)
        articles_dir = os.path.join(ds_dir, self.articles_dir)
        if not os.path.isdir(articles_dir):
            os.makedirs(articles_dir)
            
        
        for claim_metadata in tqdm(self.claims_metadata):
            response = get_request(claim_metadata['review_url'], self.retries)
            
            if response is None:
                print('Warning! Could not retrieve the review article! Skipping...')
                continue
            
            file_path = os.path.join(articles_dir, f"review_{claim_metadata['id']}.html")
            print(f'Saving review article: {file_path}')
            with open(file_path, 'w') as f_obj:
                f_obj.write(response.text)
                  
            relevant_urls = self.extractRelevantUrls(response.text)
            
            claim_metadata['relevant_urls'] = relevant_urls
            
            
        self.saveClaimsMetadata(with_relevant=True)
        
