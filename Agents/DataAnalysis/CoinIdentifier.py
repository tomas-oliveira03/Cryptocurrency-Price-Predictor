import asyncio
import os
import jsonpickle
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from Services.DataAnalysis.CoinIndentifier import CoinIdentifier
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[38;2;75;0;130m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class CoinIdentifierAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.coinIdentifier = CoinIdentifier(SHOW_LOGS=False)
        self.providerAgentName = "CoinIdentifier"
        self.queue = asyncio.Queue()
            

    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "data_analysis_request":
                        payload = jsonpickle.decode(msg.body)
                        await self.agent.queue.put(payload)  
                        print(f"{AGENT_NAME} Payload enqueued...")
                        
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        
        
    class ProcessingQueueBehav(CyclicBehaviour):
        async def run(self):
            # Waits until there is something in the queue
            payload = await self.agent.queue.get() 
            try:
                databaseCollectionName = payload.getDatabaseCollectionName()
                providerAgentName = payload.getProviderAgentName()

                if not databaseCollectionName or not providerAgentName:
                    print(f"{AGENT_NAME} \033[91mERROR\033[0m Invalid payload arguments.")
                    return

                print(f"{AGENT_NAME} Analyzing new data for {databaseCollectionName}...")
                self.agent.coinIdentifier.idetifyCoinsForAllCollections(databaseCollectionName)

                payload.setProviderAgentName(self.agent.providerAgentName)

                print(f"{AGENT_NAME} Redirecting message back to Data Analysis Orchestrator...")
                await sendMessage(self, "dataAnalysisOrchestrator", "data_analysis_finished", payload)

            except Exception as e:
                print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                
            finally:
                self.agent.queue.task_done()


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        self.add_behaviour(self.ProcessingQueueBehav())
        