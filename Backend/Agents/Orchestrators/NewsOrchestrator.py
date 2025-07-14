import os
import jsonpickle
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[31m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class NewsOrchestratorAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.providerAgentName = "NewsOrchestrator"
            
            
    class NotifyNewsSpecialists(OneShotBehaviour):
        async def run(self):
            
            print(f"{AGENT_NAME} Notifying Reddit Posts Scraper Agent to start...")
            await sendMessage(self, "redditPosts", "start_agent")
            
            print(f"{AGENT_NAME} Notifying Articles Scraper Agent to start...") 
            await sendMessage(self, "articlePosts", "start_agent")
            
            print(f"{AGENT_NAME} Notifying Forum Scraper Agent to start...") 
            await sendMessage(self, "forumPosts", "start_agent")
            
            
    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        self.agent.add_behaviour(self.agent.NotifyNewsSpecialists())
                        
                    case "job_finished":
                        payload = jsonpickle.decode(msg.body)
                        databaseCollectionName = payload.getDatabaseCollectionName()
                        
                        if not databaseCollectionName:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria. Invalid payload arguments.")
                            return
                        
                        payload.setProviderAgentName(self.agent.providerAgentName)
                        
                        print(f"{AGENT_NAME} Redirecting message to Data Analysis Orchestrator...")
                        await sendMessage(self, "dataAnalysisOrchestrator", "new_data_to_analyze", payload)

                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        