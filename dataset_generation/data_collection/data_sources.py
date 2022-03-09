import time
import datetime
import json
import sys
import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import get_request
from data_source import DataSource
from google_fact_check_ds import GoogleFactCheckDS

import requests


class PolitifactDS(DataSource):
    
    def __init__(self, data_source, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
        
    
    def retrieveClaimsMetadata(self):
        
        all_claims_metadata = []
        
            
        print('Retrieving claims from politifact...')
        
        get_params = {'page': 1}
        
        while True:
            r = get_request(self.claim_search_endpoint, self.retries, request_params=get_params)
            
            if r is None:
                print(f"Finished retrieving claims. Retrieved {get_params['page']-1} pages.")
                break
            
            claims_metadata, last_page = self.extractClaimsMetadata(r.text, get_params['page'])
            print(f"Retrieved {len(claims_metadata)} claims from page {get_params['page']}")
            all_claims_metadata.extend(claims_metadata)
            
            if last_page == True:
                print(f"Finished retrieving claims. Retrieved {get_params['page']} pages.")
                break
            
            get_params['page'] += 1
            
            #time.sleep(5)
            sys.stdout.flush()
        
        self.claims_metadata = all_claims_metadata
        
        total_claims = len(all_claims_metadata)
        
        print(f'Total claims retrieved: {total_claims}')
        
        print('Saving claims...')
        self.saveClaimsMetadata()
        
        
    def extractClaimsMetadata(self, page, page_num):
        
        month_abv_to_num = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        soup = BeautifulSoup(page, 'html.parser')
        
        links = soup.select("a.c-button.c-button--hollow")
        last_page = True
        next_page_url = f"?page={page_num+1}&".format(page_num=page_num)
        for link in links:
            if link.attrs["href"] == next_page_url:
                last_page = False
                break
        
        container_node = soup.find('section', class_='o-listicle')
        
        if container_node is None:
            print('Warning! Could not find the claims container!')
            return [], last_page
        
        claim_nodes = container_node.find_all('li', class_='o-listicle__item')
        
        if len(claim_nodes) == 0:
            print('Warning! found 0 claims.')
            return [], last_page
        
        claims_metadata = []
        for claim_node in claim_nodes:
            claim_metadata = {}
            
            name_node = claim_node.find('a', class_='m-statement__name')
            
            claim_metadata['claimant'] = name_node.text.strip()
            
            quote_node = claim_node.find('div', class_='m-statement__quote')
            
            quote_link_node = quote_node.find('a')
            
            claim_metadata['claim'] = quote_link_node.text.strip()
            
            claim_metadata['review_url'] = 'https://www.politifact.com' + quote_link_node.attrs['href']
            
            claim_date_node = claim_node.find('div', class_='m-statement__desc')
            
            if claim_date_node is not None:
                date_text = claim_date_node.text.lower()
                date_text = date_text[date_text.find('stated on')+len('stated on'):].strip()
                
                match_obj = re.search(r'\d{4}', date_text)
                if match_obj is not None:
                    claim_metadata['claim_date'] = datetime.datetime.strptime(date_text[:match_obj.start()+4], '%B %d, %Y').date().isoformat()
                else:
                    claim_metadata['claim_date'] = None
            else:
                clam_metadata['claim_date'] = None
                
            claim_metadata["reviewer_name"] = "PolitiFact"
            claim_metadata["reviewer_site"] = "politifact.com"
            
            stmnt_mtr_node = claim_node.find('div', class_='m-statement__meter')
            
            rating_img_node = stmnt_mtr_node.find('img', class_='c-image__original')
            assert rating_img_node is not None, 'Error! Could not find rating image!'
            
            claim_metadata['rating'] = rating_img_node.attrs['alt']
            
            review_url_parts = claim_metadata['review_url'].split('/')
            claim_metadata['review_date'] = datetime.date(int(review_url_parts[4]), month_abv_to_num[review_url_parts[5]], int(review_url_parts[6])).isoformat()
            
            claims_metadata.append(claim_metadata)
        
        return claims_metadata, last_page
            
        
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
                print(f"Warning! Could not retrieve the review article: {claim_metadata['review_url']}! Skipping...")
                continue
            
            with open(os.path.join(articles_dir, f"review_{claim_metadata['id']}.html"), 'w') as f_obj:
                f_obj.write(response.text)
                  
            relevant_urls = self.extractRelevantUrls(response.text)
            
            claim_metadata['relevant_urls'] = relevant_urls
                         
            
        self.saveClaimsMetadata(with_relevant=True)
            
            
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_node = soup.find('section', id='sources')
        if sources_node is None:
            return []
        
        link_nodes = sources_node.find_all('a')
        relevant_urls = []
        for link_node in link_nodes:
            try:
                relevant_urls.append(link_node.attrs['href'])
            except KeyError as e:
                continue
            
        return relevant_urls 



class SnopesDS(DataSource):
    
    def __init__(self, data_source, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
        
    
    def retrieveClaimsMetadata(self):
        
        self.combinedMetadataReviewRetrieval()
        
        
    def retrieveReviewArticle(self):
        
        self.combinedMetadataReviewRetrieval()
        
        
    def combinedMetadataReviewRetrieval(self):
                
        print('Note: Snopes.com data source combines the metadata and review retrieval into a single operation invocable by using either argument.')
        sys.stdout.flush()
        
        ds_dir = os.path.join(self.data_dir, self.data_source)
        if not os.path.isdir(ds_dir):
            os.makedirs(ds_dir)
        
        review_file_path = os.path.join(ds_dir, 'review_urls.json')
        if os.path.isfile(review_file_path):
            print('Found review urls file. Loading review urls...')
            with open(review_file_path) as f_obj:
                review_urls = json.load(f_obj)
            print(f'Loaded {len(review_urls)} review urls.')
        else:
            print('Retrieving review urls from snopes...')
            sys.stdout.flush()
            review_urls = []
            
            page = 1
            
            while True:
                request_url = self.claim_search_endpoint + f'page/{page}'
                print(f'Processing URL: {request_url}')
                sys.stdout.flush()
                r = get_request(request_url, self.retries)
                
                if r is None:
                    print(f'Finished retrieving review urls. Retrieved {page-1} pages.')
                    sys.stdout.flush()
                    break
                
                review_urls.extend(self.extractReviewURLs(r.text))
                page += 1
        
            print(f'Retrieved {len(review_urls)} review urls.')
            
            with open(review_file_path, 'w') as f_obj:
                json.dump(review_urls, f_obj)
        
        all_claims_metadata = []
        
        print('Retrieving claims metadata and relevant urls from snopes...')
        sys.stdout.flush()
        
        articles_dir = os.path.join(ds_dir, self.articles_dir)
        if not os.path.isdir(articles_dir):
            os.makedirs(articles_dir)
        
        total_reviews = len(review_urls)
        curr_id = 1
        for index, review_url in enumerate(review_urls):
            print(f'Processing URL #{index+1}/{total_reviews}: {review_url}')
            
            r = get_request(review_url, self.retries)
            
            if r is None:
                print('Warning! Could not retrieve the review article web page! Skipping...')
                continue
            
            claim_metadata = self.extractMetadataRelevantURLs(r.text, review_url)
            
            if claim_metadata is None:
                print('Warning! Could not extract claim metadata and relevant urls from the review article! Skipping...')
                continue
            
            claim_metadata['id'] = curr_id
            curr_id += 1
            
            all_claims_metadata.append(claim_metadata)
            
            review_article_file_path = os.path.join(articles_dir, f"review_{claim_metadata['id']}.html")
            with open(review_article_file_path, 'w') as f_obj:
                f_obj.write(r.text)
            
            #time.sleep(5)
            sys.stdout.flush()
            
        
        self.claims_metadata = all_claims_metadata
        
        total_claims = len(self.claims_metadata)
        
        print(f'Total claims retrieved: {total_claims}')
        
        self.saveClaimsMetadata()
        self.saveClaimsMetadata(with_relevant=True)
        
    
    def extractReviewURLs(self, page):
        
        soup = BeautifulSoup(page, 'html.parser')
        
        container_node = soup.find('div', class_='card list-archive')
        
        if container_node is None:
            print('Warning! could not find the container node!')
            return []
        
        article_nodes = container_node.find_all('article', class_='fact_check')
        if len(article_nodes) == 0:
            print('Warning! found zero articles!')
            return []
        
        review_urls = []
        for article_node in article_nodes:
            link_node = article_node.find('a')
            if link_node is None:
                continue
            
            review_urls.append(link_node.attrs['href'])
        
        if len(review_urls) == 0:
            print('Warning! found zero review URLs!')
        
        return review_urls    
    
    
    def extractMetadataRelevantURLs(self, page, review_url):
        
        soup = BeautifulSoup(page, 'html.parser')
        
        container_nodes = soup.select('article.fact_check.single-main')
        
        if len(container_nodes) == 0:
            print('Warning! Could not find the claims container!')
            return None
        
        container_node = container_nodes[0]
        
        claim_node = container_node.find('div', class_='claim-text')
        
        if claim_node is None:
            print('Warning! Could not find the claim node.')
            return None
        
        claim_metadata = {
            'review_url': review_url,
            'reviewer_name': 'Snopes',
            'reviewer_site': 'snopes.com',
            'claim_date': None,
            'claimant': None
        }
        
        date_node = container_node.find('time')
        
        if date_node is None:
            print('Warning! Could not find the review date node!')
            return None
        
        claim_metadata['review_date'] = date_node.attrs["datetime"]
        
        claim_metadata['claim'] = claim_node.text.strip()
        
        rating_nodes = container_node.select('span.h3')
        
        if len(rating_nodes) == 0:
            print('Warning! Could not find the rating node!')
            return None
        
        rating_node = rating_nodes[0]
        
        claim_metadata['rating'] = rating_node.text.strip().lower()
        
        content_nodes = container_node.select('div.single-body')
        
        if len(content_nodes) == 0:
            print('Warning! Could not find the content node!')
            return None
        
        content_node = content_nodes[0]
        
        relevant_link_nodes = content_node.find_all('a')
        
        if len(relevant_link_nodes) == 0:
            print('Warning! Found zero relevant link nodes')
            return None
        
        relevant_urls = []
        for relevant_link_node in relevant_link_nodes:
            
            try:
                relevant_urls.append(relevant_link_node.attrs['href'])
            except Exception as e:
                continue
        
        if len(relevant_urls) == 0:
            print('Warning! Found zero relevant URLs!')
            return None
        
        claim_metadata['relevant_urls'] = relevant_urls
        
        return claim_metadata
    


class AfricaCheckDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
        
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_node = soup.select("#block-mainpagecontent > article > div > div.cell.medium-8.article--main")
        if len(sources_node) == 0:
            return []
        
        sources_node = sources_node[0]
        
        link_nodes = sources_node.find_all('a')
        relevant_urls = []
        for link_node in link_nodes:
            try:
                relevant_urls.append(link_node.attrs['href'])
            except KeyError as e:
                continue
        
        return relevant_urls 
    


class FactCheckDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_node = soup.find('div', class_="entry-content")
        if sources_node is None:
            return []
        
        link_nodes = sources_node.find_all('a')
        relevant_urls = []
        for link_node in link_nodes:
            try:
                relevant_urls.append(link_node.attrs['href'])
            except KeyError as e:
                continue
        
        return relevant_urls 
    


class AFPDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_nodes = soup.find_all('div', class_="article-entry")
        
        assert len(sources_nodes) <= 1, f'Error! Found more than one sources node. Total found: {len(sources_nodes)}!'
        
        if len(sources_nodes) == 0:
            return []
        else:
            sources_node = sources_nodes[0]
        
        link_nodes = sources_node.find_all('a')
        relevant_urls = []
        for link_node in link_nodes:
            try:
                relevant_urls.append(link_node.attrs['href'])
            except KeyError as e:
                continue
        
        return relevant_urls 


class BBCDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_nodes = soup.select('#main-content > div.ssrcss-6gq9s0-Wrap.e42f8510 > div > div.ssrcss-rgov1k-MainColumn.e1sbfw0p0 > article')
        
        if len(sources_nodes) > 1:
            print(f'Warning! Found more than one sources node. Total found: {len(sources_nodes)}!')
        
        if len(sources_nodes) == 0:
            return []
        else:
            sources_node = sources_nodes[0]
        
        relevant_urls = []        
        skip_nodes_data_component = ['tag-list', 'see-alsos']
        for child_node in sources_node.contents:
            if child_node.name is None:
                continue
            
            if child_node.name.lower() == 'script':
                continue
            
            if hasattr(child_node, 'attrs'):
                if 'data-component' in child_node.attrs and child_node.attrs['data-component'] in skip_nodes_data_component:
                    continue            
            
            link_nodes = child_node.find_all('a')
            
            for link_node in link_nodes:
                try:
                    relevant_urls.append(link_node.attrs['href'])
                except KeyError as e:
                    continue
        
        delete_links = [
            "https://twitter.com/bbcrealitycheck",
            "http://www.bbc.co.uk/news/uk-41928747",
            "http://www.bbc.co.uk/realitycheck"
        ]
        
        for delete_link in delete_links:
            try:
                relevant_urls.remove(delete_link)
            except ValueError:
                pass
        
        return relevant_urls 
    


class NYTimesDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_nodes = soup.select('section[name="articleBody"]')
        
        if len(sources_nodes) > 1:
            print(f'Warning! Found more than one sources node. Total found: {len(sources_nodes)}!')
        
        if len(sources_nodes) == 0:
            print(f'Warning! Found no sources node!')
            return []
        else:
            sources_node = sources_nodes[0]
        
        link_nodes = sources_node.find_all('a')
        relevant_urls = []
        for link_node in link_nodes:
            try:
                relevant_urls.append(link_node.attrs['href'])
            except KeyError as e:
                continue
        
        return relevant_urls 



class USATodayDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_nodes = soup.select('article ul')
        
        if len(sources_nodes) > 1:
            print(f'Warning! Found more than one sources node. Total found: {len(sources_nodes)}!')
        
        if len(sources_nodes) == 0:
            print(f'Warning! Found no sources node!')
            return []
        else:
            sources_node = sources_nodes[0]
        
        link_nodes = sources_node.find_all('a')
        relevant_urls = []
        for link_node in link_nodes:
            try:
                relevant_urls.append(link_node.attrs['href'])
            except KeyError as e:
                continue
        
        return relevant_urls 



class AltNewsDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_nodes = soup.select('#main > div.m-hentry-s.m-h-s.post > div > div')
        
        if len(sources_nodes) > 1:
            print(f'Warning! Found more than one sources node. Total found: {len(sources_nodes)}!')
        
        if len(sources_nodes) == 0:
            return []
        else:
            sources_node = sources_nodes[0]
        
        relevant_urls = []
        skip_nodes_id = ['jp-relatedposts', 'enhancedtextwidget-12']
        skip_nodes_class = set(['twitter-tweet', 'slideshow-window', 'embed-youtube', 'fb-video'])
        for child_node in sources_node.contents:
            if child_node.name is None:
                continue
            
            if child_node.name.lower() == 'script':
                continue
            
            if hasattr(child_node, 'attrs'):
                if 'id' in child_node.attrs and child_node.attrs['id'] in skip_nodes_id:
                    continue
                
                if 'class' in child_node.attrs:
                    if type(child_node.attrs['class']) == list:
                        class_set = set(child_node.attrs['class'])
                        if len(class_set.intersection(skip_nodes_class)) > 0:
                            continue
                    elif child_node.attrs['class'] in skip_nodes_class:
                        continue
            
            
            link_nodes = child_node.find_all('a')
            
            for link_node in link_nodes:
                try:
                    relevant_urls.append(link_node.attrs['href'])
                except KeyError as e:
                    continue
        
        return relevant_urls 



class FullFactDS(GoogleFactCheckDS):
    
    def __init__(self, data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries):
        
        super().__init__(data_source, publisher_site, api_key, claim_search_endpoint, data_dir, articles_dir, claim_metadata_file, claim_metadata_file_w_relevant, retries)
    
    
    def extractRelevantUrls(self, review_article):
        
        soup = BeautifulSoup(review_article, 'html.parser')
        
        sources_nodes = soup.select('article')
        
        if len(sources_nodes) > 1:
            print(f'Warning! Found more than one sources node. Total found: {len(sources_nodes)}!')
        
        if len(sources_nodes) == 0:
            return []
        else:
            sources_node = sources_nodes[0]
        
        relevant_urls = []
        skip_nodes_class = set(['social-media'])
        for child_node in sources_node.contents:
            if child_node.name is None:
                continue
            
            if child_node.name.lower() == 'script':
                continue
            
            if hasattr(child_node, 'attrs'):
                if 'id' in child_node.attrs and child_node.attrs['id'] in skip_nodes_id:
                    continue
                
                if 'class' in child_node.attrs:
                    if type(child_node.attrs['class']) == list:
                        class_set = set(child_node.attrs['class'])
                        if len(class_set.intersection(skip_nodes_class)) > 0:
                            continue
                    elif child_node.attrs['class'] in skip_nodes_class:
                        continue
            
            
            link_nodes = child_node.find_all('a')
            
            for link_node in link_nodes:
                try:
                    relevant_urls.append(link_node.attrs['href'])
                except KeyError as e:
                    continue
        
        return relevant_urls 




    
    
    
    
