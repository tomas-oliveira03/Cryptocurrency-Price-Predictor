import requests
import os
import datetime
from pymongo import MongoClient
from bson.codec_options import CodecOptions

class RedditScraper:
    
    def __init__(self):
        self.redditClientID = os.getenv("REDDIT_CLIENT_ID")
        self.redditSecretKey = os.getenv("REDDIT_SECRET_KEY")
        self.redditUsername = os.getenv("REDDIT_USERNAME")
        self.redditPassword = os.getenv("REDDIT_PASSWORD")
        mongoDBURI = os.getenv("MONGODB_URI")
        
        if not mongoDBURI:
            raise ValueError("Please set the mongoDB environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.mongoCollection = mongoClient['reddit'].get_collection('posts', codec_options=CodecOptions(tz_aware=True))

        if not self.redditClientID or not self.redditSecretKey or not self.redditUsername or not self.redditPassword:
            raise ValueError("Please set the Reddit environment variables first.")
        
        
        # Create Reddit API connection
        self.headers=self.connectToReddit()
        
        self.allSubreddits=self.getAllSubreddits()
    
    
    def connectToReddit(self):
        auth = requests.auth.HTTPBasicAuth(self.redditClientID, self.redditSecretKey)
        data = {
            'grant_type': 'password',
            'username': self.redditUsername,
            'password': self.redditPassword
        }
        headers = {'User-Agent': 'API/v1'}
        res = requests.post('https://www.reddit.com/api/v1/access_token',auth=auth, data=data, headers=headers)
        token=res.json()['access_token']
        headers['Authorization'] = f"bearer {token}"
        return headers
        
        
    def getSubredditPosts(self, subreddit, sort, limit, after):
        params = {'limit': limit}
        if after:
            params['after'] = after
            
        url = f"https://oauth.reddit.com/r/{subreddit}/{sort}"
        
        res = requests.get(url, headers=self.headers, params=params)
        
        if res.status_code != 200:
            raise Exception(f"Error fetching posts error: {res.status_code} on subreddit {subreddit}")
        
        return res.json()['data']


    def processSubredditData(self, subreddit, sort="hot", pages=25, limit=100, after=None):            
        
        for page in range(pages):
            addedPosts = 0
            
            params = {'limit': limit}
            if after:
                params['after'] = after
            
            data = self.getSubredditPosts(subreddit, sort, limit, after)
            posts = data['children']
            
            if not posts:
                print("No more posts.")
                break
            
            for post in posts:
                data = post['data']
                created_date = datetime.datetime.fromtimestamp(data['created_utc'], tz=datetime.timezone.utc)

                redditPostId=post['kind']+'_'+post['data']['id']
                
                after=redditPostId
                
                # Check if post was already processed, if it is ignore it
                if self.mongoCollection.find_one({'id': redditPostId, 'subreddit': subreddit}):
                    continue
            
                addedPosts += 1
                
                post_document = {
                    'id': post['kind'] + '_' + data['id'],
                    'subreddit': subreddit,
                    'title': self.cleanText(data['title']),
                    'text': self.cleanText(data['selftext']),
                    'score': data['score'],
                    'created_at': created_date,
                    'num_comments': data['num_comments'],
                    'scraped_at': datetime.datetime.now(datetime.timezone.utc)
                }

                self.mongoCollection.insert_one(post_document)

            print(f"Page {page+1} done. Inserted {addedPosts} posts.")
            if not after:
                print("No more pages.")
                break
        
        
    def processAllSubreddits(self):
        for subreddit in self.allSubreddits:
            
            print("Scraping subreddit: ", subreddit)
            self.processSubredditData(subreddit=subreddit)
            print("=====\n\n")
        
        
        
    # Utils
    def getAllSubreddits(self):
        with open('utils/allSubreddits.txt', 'r') as file:
            subreddits = [line.strip() for line in file.readlines()]
        return subreddits
  
    def cleanText(self, text):
        return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
