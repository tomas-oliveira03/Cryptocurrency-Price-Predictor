from News import Reddit
from dotenv import load_dotenv


load_dotenv()

redditScraper = Reddit.RedditScraper()

redditScraper.processAllSubreddits()