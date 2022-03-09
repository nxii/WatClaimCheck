from abc import ABC
import os
import json
from utils import parallel_retrieval

class DataSource(ABC):
    
    def __init__(self, data_source, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        self.data_source = data_source
        self.claim_search_endpoint = claim_search_endpoint
        self.data_dir = data_dir
        self.articles_dir = articles_dir
        self.claim_metadata_file = claim_metadata_file
        self.claim_metadata_file_w_relevant = claim_metadata_file_w_relevant
        self.retries = retries
    
    
    @staticmethod 
    def retrieveClaimsMetadata(self):
        pass
    
    
    @staticmethod
    def retrieveReviewArticle(self):
        pass
    
    
    def saveClaimsMetadata(self, with_relevant=False):
        
        ds_dir = os.path.join(self.data_dir, self.data_source)
        if not os.path.isdir(ds_dir):
            os.makedirs(ds_dir)
        
        if with_relevant:
            file_path = os.path.join(ds_dir, self.claim_metadata_file_w_relevant)
        else:
            file_path = os.path.join(ds_dir, self.claim_metadata_file)
            
        with open(file_path, 'w') as f_obj:
            json.dump(self.claims_metadata, f_obj, indent=4)
            
    
    def loadClaimsMetadata(self, with_relevant=False):
        ds_dir = os.path.join(self.data_dir, self.data_source)
        
        if with_relevant:
            file_path = os.path.join(ds_dir, self.claim_metadata_file_w_relevant)
        else:
            file_path = os.path.join(ds_dir, self.claim_metadata_file)
        
        with open(file_path, 'r') as f_obj:
            self.claims_metadata = json.load(f_obj)
        
    
    def retrieveRelevantArticles(self):
        
        dir_path = os.path.join(self.data_dir, self.data_source,  self.articles_dir)
        
        relevant_urls = []
        for claim_metadata in self.claims_metadata:
            if 'relevant_urls' not in claim_metadata:
                continue
            
            for index, relevant_url in enumerate(claim_metadata['relevant_urls']):
                file_path = os.path.join(dir_path, f"relevant_{claim_metadata['id']}_{index+1}.html")
                
                relevant_urls.append({'url': relevant_url, 'file_path': file_path})
        
        response = parallel_retrieval(relevant_urls)   
