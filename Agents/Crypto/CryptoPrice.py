import asyncio
import os
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from Communication.PriceAlert import PriceAlert
from Communication.InformJobEnded import InformJobEnded
from Services.Crypto.CryptoPrice import CryptoPrice
from Agents.utils.messageHandler import sendMessage
from Agents.utils.cron import CronExpression, getSecondsUntilNextAlignedMark

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[38;5;198m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class CryptoPriceAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.cryptoPrice = CryptoPrice(SHOW_LOGS=False)
        self.isJobRunning = False
        self.informJobEnded = InformJobEnded("crypto-price")
        self.priceAlert = PriceAlert("REAL")
            

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
                            periodicJobBehavior = self.agent.PeriodicPriceCheck(period=CronExpression.EVERY_MINUTE.value)
                            self.agent.add_behaviour(periodicJobBehavior)
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    class PeriodicPriceCheck(PeriodicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Running periodic crypto price...")
            try:
                loop = asyncio.get_event_loop()
                coinsInfo = await loop.run_in_executor(None, self.agent.cryptoPrice.fetchTopCoinsPrices)
                
                print(f"{AGENT_NAME} Got coins pricing information, notifying CryptoOrchestrator...")
                await sendMessage(self, "cryptoOrchestrator", "job_finished", self.agent.informJobEnded)
                
                print(f"{AGENT_NAME} Sending coin information to NotifierAgent...")
                self.agent.priceAlert.setAllCryptoPrices(coinsInfo)
                await sendMessage(self, "notificationsAgent", "price_alert", self.agent.priceAlert)
                self.agent.priceAlert.clearCryptoPrices()
                
            except Exception as e:
                print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                return


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        