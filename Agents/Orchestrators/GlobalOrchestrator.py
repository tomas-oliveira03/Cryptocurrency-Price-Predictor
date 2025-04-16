import os
import jsonpickle
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
            
            print(f"{AGENT_NAME} Notifying DataAnalysisOrchestrator Agent to start...")
            await sendMessage(self, "dataAnalysisOrchestrator", "start_agent")
            
            
    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "new_data_available":
                        payload = jsonpickle.decode(msg.body)
                        databaseCollectionName = payload.getDatabaseCollectionName()
                        providerAgentName = payload.getProviderAgentName()
                        
                        if not databaseCollectionName or not providerAgentName:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria. Invalid payload arguments.")
                            return
                        
                        print(f"{AGENT_NAME} New data available to send to prediction model. {payload.toString()}")
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.NotifyOrchestratorSpecialists())
        self.add_behaviour(self.ReceiveRequestBehav())
        