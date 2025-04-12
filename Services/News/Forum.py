from dotenv import load_dotenv
import requests
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

class Forum:
    
    def __init__(self):
        load_dotenv()
        self.cryptoPanicAPIKey = os.getenv("CRYPTOPANIC_API_KEY")
        if not self.cryptoPanicAPIKey:
            raise ValueError("Please set cryptoPanicAPIKey environment variable first.")
        
        self.base_url = "https://cryptopanic.com/api/v1"
        self.headers = {
            'User-Agent': 'Python CryptoPanic API Client',
            'Accept': 'application/json'
        }
    
    def getPosts(self, 
                 currencies: Optional[List[str]] = None, # BTC|ETH|SOL
                 filter_type: Optional[str] = None, # rising|hot|bullish|bearish|important|saved|lol
                 kind: Optional[str] = None, # news|media
                 limit: int = 50) -> Dict:
        
        params = {"auth_token": self.cryptoPanicAPIKey, "limit": limit}
            
        if currencies:
            params["currencies"] = ",".join(currencies)
            
        if filter_type:
            params["filter"] = filter_type
            
        if kind:
            params["kind"] = kind

        
        try:
            response = requests.get(f"{self.base_url}/posts/", params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            if response and response.text:
                print(f"Response text: {response.text}")
            return {"error": str(e)}
    
    
    def save_to_json(self, data: Dict, filename: Optional[str] = None) -> str:
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'cryptopanic_data_{timestamp}.json'
        
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        # Process the data before saving
        if 'results' in data and isinstance(data['results'], list):
            for result in data['results']:
                # 1. Remove "kind" entry
                if 'kind' in result:
                    del result['kind']
                    
                if 'domain' in result:
                    del result['domain']
                    
                if 'published_at' in result:
                    del result['published_at']
                
                if 'slug' in result:
                    del result['slug']
                
                # Simplify currencies to only include code and name
                if 'currencies' in result and isinstance(result['currencies'], list):
                    simplified_currencies = []
                    for currency in result['currencies']:
                        if isinstance(currency, dict) and 'code' in currency and 'title' in currency:
                            simplified_currencies.append({
                                'code': currency['code'],
                                'name': currency['title']
                            })
                    result['currencies'] = simplified_currencies
                
                # 2. Make title and published_at the first entries by creating a new ordered dict
                new_result = {}
                
                # Add id first if it exists
                if 'id' in result:
                    new_result['id'] = result['id']
                    
                # Add title first if it exists
                if 'title' in result:
                    new_result['title'] = result['title']
                
                # Add published_at second if it exists
                if 'created_at' in result:
                    new_result['created_at'] = result['created_at']
                
                if 'url' in result:
                    new_result['url'] = result['url']
                
                
                # 3. Replace source object with just its title
                if 'source' in result and isinstance(result['source'], dict) and 'title' in result['source']:
                    source_title = result['source']['title']
                    result['source'] = source_title
                
                # Add all remaining fields (excluding title and published_at which we already added)
                for key, value in result.items():
                    if key not in ['title', 'published_at'] and key not in new_result:
                        new_result[key] = value
                
                # Replace the original result with our reordered one
                result.clear()
                result.update(new_result)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(data["results"], f, ensure_ascii=False, indent=4)
        
        print(f"Data saved to {filepath}")
        return filepath


if __name__ == "__main__":

    # Initialize the API client with your API key
    api = Forum()
    
    # Example 1: Get latest news (default)
    print("\n1. Getting latest news...\n")
    latest_news = api.getPosts()
    # api.save_to_json(latest_news, "bullish_btc_news.json")
    # print("Latest news", len(latest_news["results"]))
    
    # Example 2: Filter by currencies
    print("\n2. Getting BTC and ETH news...\n")
    
    listOfCurrencies = ['BTC', 'ETH']
    for curr in listOfCurrencies:
        crypto_news = api.getPosts(currencies=curr)
        api.save_to_json(crypto_news, "bullish_btc_news.json")
        print("Latest news", len(crypto_news["results"]))
    
    # Example 3: Filter by rising news
    print("\n3. Getting rising news...\n")
    # rising_news = api.getPosts(filter_type="rising")
    # api.save_to_json(rising_news, "bullish_btc_news.json")
    # print("Latest news", len(rising_news))
    
    # Example 4: Save results to file
    print("\n4. Saving bullish BTC news to file...\n")
    # bullish_btc = api.getPosts(currencies=["BTC"], filter_type="bullish")
    # api.save_to_json(bullish_btc, "bullish_btc_news.json")

