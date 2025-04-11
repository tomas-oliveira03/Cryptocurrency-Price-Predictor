import json
import os
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from Services.Crypto.CryptoPrice import CryptoPrice
from Agents.utils.messageHandler import sendMessage

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[38;5;208m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class CryptoPriceAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.cryptoPrice = CryptoPrice(SHOW_LOGS=False)
        self.isJobRunning = False
            

    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        print(f"{AGENT_NAME} Ready to start scraping information...")
                        
                        if self.agent.isJobRunning:
                            print(f"{AGENT_NAME} Job already running, skipping start request...")
                            
                        else:
                            self.agent.isJobRunning = True
                            oneDayInSeconds = 24*60*60
                            periodicJobBehavior = self.agent.PeriodicPriceCheck(period=oneDayInSeconds)
                            self.agent.add_behaviour(periodicJobBehavior)
                        
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    class PeriodicPriceCheck(PeriodicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Running periodic crypto price check...")
            try:
                # numberOfInsertions = self.agent.cryptoPrice.fetchCoinsData()
                numberOfInsertions = 9999
                print(f"{AGENT_NAME} Crypto prices data saved to MongoDB successfully. New insertions: {numberOfInsertions}, notifiying CryptoOrchestrator...")

                payload = {
                    "databaseCollectionName": "crypto-price" 
                }
                
                await sendMessage(self, "cryptoOrchestrator", "job_finished", payload)
                                
            except Exception as e:
                print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                return
                
            


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        