import json
import os
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[32m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class SentimentAnalysisAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
            

    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        print(f"{AGENT_NAME} Ready to receive information...")
                        
                    case "new_data_to_analyze":
                        payload = json.loads(msg.body)
                        mongoDBCollection = payload.get("databaseCollectionName")
                        
                        if not mongoDBCollection:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria.")
                            return
                        
                        # Analyze data
                        print(f"{AGENT_NAME} collection", mongoDBCollection)
                        print(f"{AGENT_NAME} analyze data")
                        
                        # Send message
                        payload["providerName"] = "SentimentAnalysis"
                        await sendMessage(self, "globalOrchestrator", "new_data_available", payload)
                        
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        



    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        