import asyncio
import os
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
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
                            
                            # delay = getSecondsUntilNextAlignedMark(CronExpression.EVERY_10_MINUTES)
                            # print(f"Waiting {delay} seconds to align with next 10-minute mark.")
                            # await asyncio.sleep(delay)
                            
                            periodicJobBehavior = self.agent.PeriodicPriceCheck(period=CronExpression.EVERY_10_MINUTES.value)
                            self.agent.add_behaviour(periodicJobBehavior)
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    class PeriodicPriceCheck(PeriodicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Running periodic crypto price...")
            try:
                loop = asyncio.get_event_loop()
                # numberOfInsertions = await loop.run_in_executor(None, self.agent.cryptoPrice.fetchTopCoinsPrices)
                # print(f"{AGENT_NAME} Crypto price data saved to MongoDB successfully. New insertions: {numberOfInsertions}, notifying CryptoOrchestrator...")
                
                await sendMessage(self, "cryptoOrchestrator", "job_finished", self.agent.informJobEnded)
                                
            except Exception as e:
                print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                return


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        