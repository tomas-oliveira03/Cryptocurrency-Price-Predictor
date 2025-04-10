import asyncio
import spade
import signal
import asyncio
import os
from dotenv import load_dotenv
import spade
from Orchestrators.MainOrchestrator import MainOrchestrator
from Orchestrators.CryptoOrchestrator import CryptoOrchestrator
from Orchestrators.NewsOrchestrator import NewsOrchestrator




async def main():
    # Your Spade agent startup code
    load_dotenv()
    SPADE_DOMAIN = os.getenv("SPADE_DOMAIN")
    SPADE_PASSWORD = os.getenv("SPADE_PASSWORD")
    
    if not SPADE_DOMAIN or not SPADE_PASSWORD:
        raise ValueError("Please set the SPADE_DOMAIN and SPADE_PASSWORD environment variables first.")
    
    # Start agents
    mainOrchestratorAgent = MainOrchestrator(f"mainOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    cryptoOrchestratorAgent = CryptoOrchestrator(f"cryptoOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    newsOrchestratorAgent = NewsOrchestrator(f"newsOrchestrator@{SPADE_DOMAIN}", SPADE_PASSWORD, SPADE_DOMAIN)
    
    
    await newsOrchestratorAgent.start(auto_register=True)
    await cryptoOrchestratorAgent.start(auto_register=True)
    await mainOrchestratorAgent.start(auto_register=True)
    

    await spade.wait_until_finished(mainOrchestratorAgent)
    await spade.wait_until_finished(cryptoOrchestratorAgent)


def signal_handler(sig, frame):
    print('Shutting down...')
    loop.stop()


if __name__ == "__main__":
    # Register signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
