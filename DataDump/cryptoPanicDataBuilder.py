import csv
import json
import os
from datetime import datetime

def read_csv(file_path):
    """Read CSV file and return list of dictionaries."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            return list(csv_reader)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def save_json(data, file_path):
    """Save data to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    print(f"JSON file saved to {file_path}")

def create_merged_json():
    # Define file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cryptopanic_news_path = os.path.join(current_dir, "cryptopanic_news.csv")
    source_path = os.path.join(current_dir, "source.csv")
    currency_path = os.path.join(current_dir, "currency.csv")
    news_currency_path = os.path.join(current_dir, "news_currency.csv")
    
    # Read CSV files
    news_data = read_csv(cryptopanic_news_path)
    source_data = read_csv(source_path)
    currency_data = read_csv(currency_path)
    news_currency_data = read_csv(news_currency_path)
    
    # Create dictionaries for faster lookups
    sources = {item.get('id'): item for item in source_data} if source_data else {}
    currencies = {item.get('id'): item for item in currency_data} if currency_data else {}
    
    # Create map of newsId to currencyIds
    news_currencies = {}
    for item in news_currency_data:
        if item.get('newsId') in news_currencies:
            news_currencies[item.get('newsId')].append(item.get('currencyId'))
        else:
            news_currencies[item.get('newsId')] = [item.get('currencyId')]
    
    # Define filter date (April 10, 2024)
    filter_date = datetime(2024, 4, 10)
    
    # Define vote fields that should be nested
    vote_fields = ['negative', 'positive', 'important', 'liked', 'disliked', 'lol', 'toxic', 'saved', 'comments']
    
    # Merge data
    merged_data = []
    for news in news_data:
        news_id = news.get('id')
        source_id = news.get('sourceId')
        
        # Filter data from 10/04/2024 forwards
        news_date_str = news.get('newsDatetime')
        if news_date_str:
            try:
                # Try to parse the date
                news_date = datetime.strptime(news_date_str, "%Y-%m-%d %H:%M:%S")
                if news_date < filter_date:
                    continue  # Skip this news item as it's before our filter date
            except ValueError:
                # If date parsing fails, include the item
                pass
        
        # Create merged news item
        merged_item = {}
        votes = {}  # Create votes object
        
        for key, value in news.items():
            # Process ID field
            if key == 'id':
                merged_item[key] = int(value) if value else None
            # Rename newsDatetime to published_at
            elif key == 'newsDatetime':
                merged_item['created_at'] = value
            # Handle vote fields - add to votes object only, not to top level
            elif key in vote_fields:
                if value is not None:
                    try:
                        votes[key] = int(value)
                    except (ValueError, TypeError):
                        votes[key] = value
            # Handle other fields (excluding sourceId and vote fields)
            elif key not in ['sourceId'] + vote_fields:
                merged_item[key] = value
        
        # Add votes object to merged item
        merged_item['votes'] = votes
        
        # Add source information - just the name
        if source_id in sources:
            merged_item['source'] = sources[source_id].get('name')
        else:
            merged_item['source'] = None
            
        # Add currency information - filtered to remove 'id' field
        merged_item['currencies'] = []
        if news_id in news_currencies:
            for currency_id in news_currencies[news_id]:
                if currency_id in currencies:
                    currency_data = currencies[currency_id]
                    filtered_currency = {
                        "code": currency_data.get('code'),
                        "name": currency_data.get('name')
                    }
                    merged_item['currencies'].append(filtered_currency)
        
        merged_data.append(merged_item)
    
    # Save to JSON file
    output_path = os.path.join(current_dir, "CryptoPanicDataDump.json")
    save_json(merged_data, output_path)
    return output_path

if __name__ == "__main__":
    output_path = create_merged_json()
    print(f"Merged data saved to: {output_path}")
