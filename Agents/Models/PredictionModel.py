import asyncio
import os
import jsonpickle
from spade.agent import Agent
from Communication.PriceAlert import PriceAlert
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
from Agents.utils.messageHandler import sendMessage
from Agents.utils.cron import CronExpression
from Services.Models.PredictionModel import PredictionModel

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[38;5;88m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class PredictionModelAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.predictionModel = PredictionModel(SHOW_LOGS=False)
        self.isJobRunning = False
        self.isFirstTime = True
        self.priceAlert = PriceAlert("PREDICTED")
        self.queue = asyncio.Queue()


    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        print(f"{AGENT_NAME} Ready to start predicting models...")
                        
                        if self.agent.isJobRunning:
                            print(f"{AGENT_NAME} Job already running, skipping start request...")
                            
                        else:
                            self.agent.isJobRunning = True
                            periodicJobBehavior = self.agent.PeriodicPredictionModel(period=CronExpression.EVERY_30_MINUTES.value)
                            self.agent.add_behaviour(periodicJobBehavior)
                            
                    case "prediction_request":
                        payload = jsonpickle.decode(msg.body)
                        await self.agent.queue.put(payload)  
                        print(f"{AGENT_NAME} Payload enqueued...")
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    class PeriodicPredictionModel(PeriodicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Running periodic prediction model...")
            
            if self.agent.isFirstTime: 
                self.agent.isFirstTime = False  
                return
            
            if self.agent.queue.empty():
                print(f"{AGENT_NAME} No new data — skipping prediction job.")
                return
            
            try:
                # Clear the queue
                while not self.agent.queue.empty():
                    try:
                        self.agent.queue.get_nowait()
                        self.agent.queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                
                loop = asyncio.get_event_loop()
                
                forcastDays=7
                initialFetchDays=365 * 2
                
                coinsInfo = await loop.run_in_executor(None, self.agent.predictionModel.runModelForEveryCrypto, forcastDays, initialFetchDays)
                print(f"{AGENT_NAME} New prediction made...")
                
                print(f"{AGENT_NAME} Sending coin information to NotifierAgent...")
                self.agent.priceAlert.setAllCryptoPrices(coinsInfo)
                await sendMessage(self, "notifierAgent", "price_alert", self.agent.priceAlert)
                self.agent.priceAlert.clearCryptoPrices()
                
                

            except Exception as e:
                print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                return


    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        