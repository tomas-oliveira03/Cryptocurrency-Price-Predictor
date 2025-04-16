import os
import jsonpickle
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[38;2;139;69;19m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class DataAnalysisOrchestratorAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.providerAgentName = "DataAnalysisOrchestrator"
            
            
    class NotifyNewsSpecialists(OneShotBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Notifying Sentiment Analysis Agent to start...")
            await sendMessage(self, "redditPosts", "sentimentAnalysis")
            
            print(f"{AGENT_NAME} Notifying Coin Extractor Agent to start...") 
            # await sendMessage(self, "articlePosts", "start_agent")
            
            
    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        self.agent.add_behaviour(self.agent.NotifyNewsSpecialists())
                        
                    case "new_data_to_analyze":
                        payload = jsonpickle.decode(msg.body)
                        databaseCollectionName = payload.getDatabaseCollectionName()
                        providerAgentName = payload.getProviderAgentName()
                        
                        if not databaseCollectionName or not providerAgentName:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria. Invalid payload arguments.")
                            return
                        
                        payload.setProviderAgentName(self.agent.providerAgentName)
                        
                        print(f"{AGENT_NAME} Redirecting message to Sentiment Analysis Agent...")
                        await sendMessage(self, "sentimentAnalysis", "data_analysis_request", payload)
                        
                    case "data_analysis_finished":
                        payload = jsonpickle.decode(msg.body)
                        payload.setProviderAgentName(self.agent.providerAgentName)
                        
                        print(f"{AGENT_NAME} Redirecting message back to Global Orchestrator...")
                        await sendMessage(self, "globalOrchestrator", "new_data_available", payload)

                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        