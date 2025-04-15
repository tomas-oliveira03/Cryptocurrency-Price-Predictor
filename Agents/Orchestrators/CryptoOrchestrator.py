import os
import jsonpickle
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[33m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class CryptoOrchestratorAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.providerAgentName = "CryptoOrchestrator"
            
            
    class NotifyCryptoSpecialists(OneShotBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Notifying DetailedCryptoData Agent to start...")
            await sendMessage(self, "detailedCryptoData", "start_agent")
            
            print(f"{AGENT_NAME} Notifying FearGreedIndex Agent to start...") 
            await sendMessage(self, "fearGreedIndex", "start_agent")
            
            print(f"{AGENT_NAME} Notifying CryptoPrice Agent to start...") 
            await sendMessage(self, "cryptoPrice", "start_agent")
            
            
    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        self.agent.add_behaviour(self.agent.NotifyCryptoSpecialists())
                        
                    case "job_finished":
                        payload = jsonpickle.decode(msg.body)
                        databaseCollectionName = payload.getDatabaseCollectionName()
                        
                        if not databaseCollectionName:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria. Invalid payload arguments.")
                            return
                        
                        payload.setProviderAgentName(self.agent.providerAgentName)
                        
                        print(f"{AGENT_NAME} Redirecting message back to Global Orchestrator...")
                        await sendMessage(self, "globalOrchestrator", "new_data_available", payload)

                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        