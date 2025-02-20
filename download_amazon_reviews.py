import os
import requests
from tqdm import tqdm
import json

# Direct download URLs for the Books category
URLS = {
    'reviews': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Books.jsonl.gz',
    'metadata': 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz'
}

def download_file(url, filename):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        print(f"\nDownloading {filename}...")
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
                
        progress_bar.close()
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    # Create data directory
    os.makedirs('amazon_books_data', exist_ok=True)
    
    # Download reviews
    if download_file(URLS['reviews'], 'amazon_books_data/books_reviews.jsonl.gz'):
        print("Successfully downloaded reviews data!")
    else:
        print("Failed to download reviews data.")
        
    # Download metadata
    if download_file(URLS['metadata'], 'amazon_books_data/books_metadata.jsonl.gz'):
        print("Successfully downloaded metadata!")
    else:
        print("Failed to download metadata.")
        
    print("\nFiles are downloaded to the 'amazon_books_data' directory.")
    print("You can use these files with the following Python code:")
    print("""
    import gzip
    import json
    
    # Read reviews
    with gzip.open('amazon_books_data/books_reviews.jsonl.gz', 'rt') as f:
        for line in f:
            review = json.loads(line.strip())
            # Process each review...
    
    # Read metadata
    with gzip.open('amazon_books_data/books_metadata.jsonl.gz', 'rt') as f:
        for line in f:
            metadata = json.loads(line.strip())
            # Process each metadata entry...
    """)

if __name__ == "__main__":
    main() 