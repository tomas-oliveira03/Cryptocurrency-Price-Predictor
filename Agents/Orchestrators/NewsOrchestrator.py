import os
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[31m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class NewsOrchestratorAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
            
            
    class NotifyNewsSpecialists(OneShotBehaviour):
        async def run(self):
            
            print(f"{AGENT_NAME} Notifying Crypto Articles Agent to start...")
            print(f"{AGENT_NAME} Notifying Crypto Reddit Agent to start...") 
            
            
    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        self.agent.add_behaviour(self.agent.NotifyNewsSpecialists())
                        
                        # TEST ONLY (CODE MISSPLACED)             
                        await sendMessage(self, "sentimentAnalysis", "new_data_to_analyze", {
                            "databaseCollectionName": "reddit"
                        })
                            
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        



    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        