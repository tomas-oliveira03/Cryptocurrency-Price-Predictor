import os
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[33m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class CryptoOrchestrator(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
            
            
    class NotifyCryptoSpecialists(OneShotBehaviour):
        async def run(self):
            
            print(f"{AGENT_NAME} Notifying Crypto Price Agent to start...")
            print(f"{AGENT_NAME} Notifying Crypto FearGreedIndex Agent to start...") 
            
            
    class ListeningForMessages(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        self.agent.add_behaviour(self.agent.NotifyCryptoSpecialists())
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        



    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ListeningForMessages())
        