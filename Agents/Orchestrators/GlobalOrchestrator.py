import os
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[34m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"


class GlobalOrchestratorAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        
    
    class ReceiveRequestBehav(OneShotBehaviour):
        async def run(self):
            
            print(f"{AGENT_NAME} Notifying CryptoOrchestrator Agent to start...")
            await sendMessage(self, "cryptoOrchestrator", "start_agent")
            
            print(f"{AGENT_NAME} Notifying NewsOrchestrator Agent to start...")
            await sendMessage(self, "newsOrchestrator", "start_agent")
            
            print(f"{AGENT_NAME} Notifying SentimentAnalysis Agent to start...")
            await sendMessage(self, "sentimentAnalysis", "start_agent")
            
            
    class ListeningForMessages(CyclicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Listening for results...")
        
        
    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        