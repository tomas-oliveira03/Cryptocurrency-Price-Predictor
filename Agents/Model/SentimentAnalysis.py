import json
import os
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from Services.Model.SentimentAnalysis import SentimentAnalysis
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[32m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class SentimentAnalysisAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.sentimentAnalysis = SentimentAnalysis(SHOW_LOGS=False)
            

    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        print(f"{AGENT_NAME} Ready to receive information...")
                        
                    case "new_data_to_analyze":
                        print(f"{AGENT_NAME} Analyzing new data...")
                        payload = json.loads(msg.body)
                        databaseCollectionName = payload.get("databaseCollectionName")
                        
                        if not databaseCollectionName:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria. Payload arguments missing.")
                            return
                        
                        # Analyze data                        
                        try:
                            self.agent.sentimentAnalysis.analyzeSentimentsForAllCollections(databaseCollectionName)
                            
                            payload["providerAgentName"] = "SentimentAnalysis"
                            await sendMessage(self, "globalOrchestrator", "new_data_available", payload)

                        except Exception as e:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                            return
                        
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        



    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        