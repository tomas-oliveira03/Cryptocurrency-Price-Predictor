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
        
        self.allAgentsNeededForTaskCompletionMap = {
            "reddit": {
                "SentimentAnalysis": False,
                "CoinIdentifier": False
            },
            "articles": {
                "SentimentAnalysis": False,
                "CoinIdentifier": False
            }
        }
            
            
    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        print(f"{AGENT_NAME} Ready to deal with data analysis information...")
                        
                    case "new_data_to_analyze":
                        payload = jsonpickle.decode(msg.body)
                        databaseCollectionName = payload.getDatabaseCollectionName()
                        providerAgentName = payload.getProviderAgentName()
                        
                        if not databaseCollectionName or not providerAgentName:
                            print(f"{AGENT_NAME} \033[91mERROR\033[0m Message does not provide intended criteria. Invalid payload arguments.")
                            return
                        
                        payload.setProviderAgentName(self.agent.providerAgentName)
                        
                        # Initialize tracking if not already present
                        if databaseCollectionName not in self.agent.allAgentsNeededForTaskCompletionMap:
                            self.agent.allAgentsNeededForTaskCompletionMap[databaseCollectionName] = {
                                "SentimentAnalysis": False,
                                "CoinIdentifier": False
                            }
                        
                        print(f"{AGENT_NAME} Redirecting message to Sentiment Analysis Agent...")
                        await sendMessage(self, "sentimentAnalysis", "data_analysis_request", payload)
                        
                        print(f"{AGENT_NAME} Redirecting message to Coin Identifier Agent...")
                        await sendMessage(self, "coinIdentifier", "data_analysis_request", payload)
                        
                    case "data_analysis_finished":
                        payload = jsonpickle.decode(msg.body)
                        
                        databaseCollectionName = payload.getDatabaseCollectionName()
                        providerAgentName = payload.getProviderAgentName()
                        
                        # Mark agent as completed
                        self.agent.allAgentsNeededForTaskCompletionMap[databaseCollectionName][providerAgentName] = True
                        
                        payload.setProviderAgentName(self.agent.providerAgentName)
                        print(f"{AGENT_NAME} Received result from {providerAgentName} for {databaseCollectionName}")
                        
                        allDone = all(self.agent.allAgentsNeededForTaskCompletionMap[databaseCollectionName].values())
                        
                        if allDone:
                            print(f"{AGENT_NAME} All analysis complete for {databaseCollectionName}. Forwarding to Global Orchestrator...")
                            payload.setProviderAgentName(self.agent.providerAgentName)
                            await sendMessage(self, "globalOrchestrator", "new_data_available", payload)

                            # Reset all agent flags to False (instead of deleting)
                            for agentName in self.agent.allAgentsNeededForTaskCompletionMap[databaseCollectionName]:
                                self.agent.allAgentsNeededForTaskCompletionMap[databaseCollectionName][agentName] = False

                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        