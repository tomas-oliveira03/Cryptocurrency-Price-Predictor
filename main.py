import asyncio
import spade
import signal
import asyncio
import os
from dotenv import load_dotenv
import spade
from Agents.Orchestrators.GlobalOrchestrator import GlobalOrchestratorAgent
from Agents.Orchestrators.CryptoOrchestrator import CryptoOrchestratorAgent
from Agents.Orchestrators.NewsOrchestrator import NewsOrchestratorAgent
from Agents.Model.SentimentAnalysis import SentimentAnalysisAgent


async def main():
    # Your Spade agent startup code
    load_dotenv()
    SPADE_DOMAIN = os.getenv("SPADE_DOMAIN")
    SPADE_PASSWORD = os.getenv("SPADE_PASSWORD")
    
    if not SPADE_DOMAIN or not SPADE_PASSWORD:
        raise ValueError("Please set the SPADE_DOMAIN and SPADE_PASSWORD environment variables first.")
    
    # Start agents
    globalOrchestratorAgent = GlobalOrchestratorAgent(f"globalOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    cryptoOrchestratorAgent = CryptoOrchestratorAgent(f"cryptoOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    newsOrchestratorAgent = NewsOrchestratorAgent(f"newsOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    sentimentAnalysisAgent = SentimentAnalysisAgent(f"sentimentAnalysis@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    
    
    await sentimentAnalysisAgent.start(auto_register=True)
    
    await newsOrchestratorAgent.start(auto_register=True)
    await cryptoOrchestratorAgent.start(auto_register=True)
    await globalOrchestratorAgent.start(auto_register=True)
    

    await spade.wait_until_finished(globalOrchestratorAgent)
    await spade.wait_until_finished(cryptoOrchestratorAgent)
    await spade.wait_until_finished(newsOrchestratorAgent)


def signal_handler(sig, frame):
    print('Shutting down...')
    loop.stop()


if __name__ == "__main__":
    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
