import os
import jsonpickle
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from Services.Models.SentimentAnalysis import SentimentAnalysis
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[32m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class SentimentAnalysisAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.sentimentAnalysis = SentimentAnalysis(SHOW_LOGS=False)
        self.providerAgentName = "SentimentAnalysis"
            

    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        print(f"{AGENT_NAME} Ready to receive information...")
                        
                    case "new_data_to_analyze":
                        payload = jsonpickle.decode(msg.body)
                        databaseCollectionName = payload.getDatabaseCollectionName()
                        providerAgentName = payload.getProviderAgentName()
                        
                        print(databaseCollectionName, providerAgentName)
                        if not databaseCollectionName or not providerAgentName:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria. Invalid payload arguments.")
                            return
                        
                        # Analyze data                        
                        print(f"{AGENT_NAME} Analyzing new data for {databaseCollectionName}...")
                        try:
                            # self.agent.sentimentAnalysis.analyzeSentimentsForAllCollections(databaseCollectionName)
                            
                            payload.setProviderAgentName(self.agent.providerAgentName)
                            
                            print(f"{AGENT_NAME} Redirecting message back to Global Orchestrator...")
                            await sendMessage(self, "globalOrchestrator", "new_data_available", payload)

                        except Exception as e:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                            return
                        
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        



    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        