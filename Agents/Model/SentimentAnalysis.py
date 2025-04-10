import os
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[32m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"

class SentimentAnalysisAgent(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
            

    class ReceiveRequestBehav(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=20)
            if msg:
                performativeReceived = msg.get_metadata("performative")
                match performativeReceived:
                    case "start_agent":
                        print(f"{AGENT_NAME} Ready to receive information...")
                
                    case _:
                        print(f"{AGENT_NAME} Invalid message performative received: {performativeReceived}")
        



    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.ReceiveRequestBehav())
        