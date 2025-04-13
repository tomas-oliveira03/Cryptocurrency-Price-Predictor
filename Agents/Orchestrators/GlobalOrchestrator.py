import json
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
        
    
    class NotifyOrchestratorSpecialists(OneShotBehaviour):
        async def run(self):
            
            print(f"{AGENT_NAME} Notifying CryptoOrchestrator Agent to start...")
            await sendMessage(self, "cryptoOrchestrator", "start_agent")
            
            print(f"{AGENT_NAME} Notifying NewsOrchestrator Agent to start...")
            await sendMessage(self, "newsOrchestrator", "start_agent")
            
            print(f"{AGENT_NAME} Notifying SentimentAnalysis Agent to start...")
            await sendMessage(self, "sentimentAnalysis", "start_agent")
            
            
    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "new_data_available":
                        payload = json.loads(msg.body)
                        print(f"{AGENT_NAME} New data available to send to prediction model. {payload}")
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.NotifyOrchestratorSpecialists())
        self.add_behaviour(self.ReceiveRequestBehav())
        