import os
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
import asyncio
from Services.News.Forum import Forum
from Agents.utils.messageHandler import sendMessage
from Agents.utils.cron import CronExpression

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[38;5;22m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class ForumAgent(Agent):
     
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        self.forumScraper = Forum(SHOW_LOGS=False)
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
                            periodicJobBehavior = self.agent.PeriodicForumPostsCheck(period=CronExpression.EVERY_DAY.value)
                            self.agent.add_behaviour(periodicJobBehavior)
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        

    class PeriodicForumPostsCheck(PeriodicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Running periodic forum scraper...")
            try:
                loop = asyncio.get_event_loop()
                # await loop.run_in_executor(None, self.agent.forumScraper.getAllInformation)
                payload = {
                    "databaseCollectionName": "forum" 
                }
                
                await sendMessage(self, "newsOrchestrator", "job_finished", payload)
                                
            except Exception as e:
                print(f"{AGENT_NAME} \033[91mERROR\033[0m {e}")
                return
                

    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        