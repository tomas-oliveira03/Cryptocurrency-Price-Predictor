import asyncio
import spade
import signal
import asyncio
import os
from dotenv import load_dotenv
import spade
from Agents.Crypto.CryptoPrice import CryptoPriceAgent
from Agents.Crypto.DetailedCryptoData import DetailedCryptoDataAgent
from Agents.Crypto.FearGreedIndex import FearGreedIndexAgent
from Agents.DataAnalysis.CoinIdentifier import CoinIdentifierAgent
from Agents.Models.PredictionModel import PredictionModelAgent
from Agents.News.Articles import ArticlesAgent
from Agents.News.Forum import ForumAgent
from Agents.News.Reddit import RedditAgent
from Agents.Notifications.Notifier import NotifierAgent
from Agents.Orchestrators.DataAnalysisOrchestrator import DataAnalysisOrchestratorAgent
from Agents.Orchestrators.GlobalOrchestrator import GlobalOrchestratorAgent
from Agents.Orchestrators.CryptoOrchestrator import CryptoOrchestratorAgent
from Agents.Orchestrators.NewsOrchestrator import NewsOrchestratorAgent
from Agents.DataAnalysis.SentimentAnalysis import SentimentAnalysisAgent

from dotenv import load_dotenv, dotenv_values
async def main():
    # Your Spade agent startup code
    load_dotenv(override=True)
    SPADE_DOMAIN = os.getenv("SPADE_DOMAIN")
    SPADE_PASSWORD = os.getenv("SPADE_PASSWORD")
    
    if not SPADE_DOMAIN or not SPADE_PASSWORD:
        raise ValueError("Please set the SPADE_DOMAIN and SPADE_PASSWORD environment variables first.")
    
    # Create agents
    globalOrchestratorAgent = GlobalOrchestratorAgent(f"globalOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    cryptoOrchestratorAgent = CryptoOrchestratorAgent(f"cryptoOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    newsOrchestratorAgent = NewsOrchestratorAgent(f"newsOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    dataAnalysisOrchestratorAgent = DataAnalysisOrchestratorAgent(f"dataAnalysisOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    detailedCryptoDataAgent = DetailedCryptoDataAgent(f"detailedCryptoData@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    fearGreedIndexAgent = FearGreedIndexAgent(f"fearGreedIndex@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    cryptoPriceAgent = CryptoPriceAgent(f"cryptoPrice@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    sentimentAnalysisAgent = SentimentAnalysisAgent(f"sentimentAnalysis@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    coinIdentifierAgent = CoinIdentifierAgent(f"coinIdentifier@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    redditAgent = RedditAgent(f"redditPosts@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    articlesAgent = ArticlesAgent(f"articlePosts@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    forumAgent = ForumAgent(f"forumPosts@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    predictionModelAgent = PredictionModelAgent(f"predictionModel@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    notifierAgent = NotifierAgent(f"notifierAgent@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    
    # Start agents
    await notifierAgent.start(auto_register=True)
    
    await predictionModelAgent.start(auto_register=True)
    
    await sentimentAnalysisAgent.start(auto_register=True)
    await coinIdentifierAgent.start(auto_register=True)
    
    await redditAgent.start(auto_register=True)
    await articlesAgent.start(auto_register=True)
    await forumAgent.start(auto_register=True)
    
    await detailedCryptoDataAgent.start(auto_register=True)
    await fearGreedIndexAgent.start(auto_register=True)
    await cryptoPriceAgent.start(auto_register=True)
    
    await newsOrchestratorAgent.start(auto_register=True)
    await cryptoOrchestratorAgent.start(auto_register=True)
    await dataAnalysisOrchestratorAgent.start(auto_register=True)
    await globalOrchestratorAgent.start(auto_register=True)
    
    
    # Wait forever 
    await spade.wait_until_finished(globalOrchestratorAgent)


def signal_handler(sig, frame):
    print('Shutting down...')
    loop.stop()


if __name__ == "__main__":
    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
