import requests
import sys
import os.path
from multiprocessing.pool import Pool


def get_request(request_url, retries, request_params=None):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
    }
            
    while retries > 0:
        
        try:
            r = requests.get(request_url, params=request_params, timeout=2, headers=headers)
        except Exception as e:
            retries -= 1
            continue
        
        if r.status_code == requests.codes.ok:
            return r
        else:
            retries -= 1 
            
    return None


def retrieve_page(retrieval_data, retries=5):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
    }
    
    
    if os.path.isfile(retrieval_data['file_path']):
        print(f"File: {retrieval_data['file_path']} already exists! Skipping...")
        sys.stdout.flush()
        return 1
    else:
        print(f'Processing: {retrieval_data}')
        sys.stdout.flush()
    
    while retries > 0:
        
        try:
            r = requests.get(retrieval_data['url'], timeout=2, headers=headers)
        except Exception as e:
            retries -= 1
            continue
        
        if r.status_code == requests.codes.ok:
            try:
                if r.headers['content-type'].find('text/html') != -1:
                    with open(retrieval_data['file_path'], 'w') as f_obj:
                        f_obj.write(r.text)
                    
                    return 1
                else:
                    return 0
            except Exception as e:
                return 0
                
        else:
            retries -= 1 
    
    return 0
    

def parallel_retrieval(data_list):
    
    with Pool(processes=8) as pool:
        responses = pool.map(retrieve_page, data_list)
    
    print(sum(responses))
