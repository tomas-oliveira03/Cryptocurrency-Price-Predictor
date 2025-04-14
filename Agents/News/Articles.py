import os
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
import asyncio
from Services.News.Articles import Articles
from Agents.utils.messageHandler import sendMessage
from Agents.utils.cron import cronExpression

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[38;5;196m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class ArticlesAgent(Agent):
     
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.articlesScraper = Articles(SHOW_LOGS=False)
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
                            periodicJobBehavior = self.agent.PeriodicArticlePostsCheck(period=cronExpression["EVERY_HOUR"])
                            self.agent.add_behaviour(periodicJobBehavior)
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    class PeriodicArticlePostsCheck(PeriodicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Running periodic articles scraper...")
            try:
                loop = asyncio.get_event_loop()
                # await loop.run_in_executor(None, self.agent.articlesScraper.fetchAllWebsiteArticlesContent)
                payload = {
                    "databaseCollectionName": "articles" 
                }
                
                await sendMessage(self, "newsOrchestrator", "job_finished", payload)
                                
            except Exception as e:
                print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                return
                

    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        