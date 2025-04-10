import os
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message

# FOR DEBUGGING ONLY
AGENT_NAME = f"\033[34m[{os.path.splitext(os.path.basename(__file__))[0]}]\033[0m"


class MainOrchestrator(Agent):
    
    def __init__(self, jid, password, spadeDomain):
        super().__init__(jid, password)
        self.spadeDomain = spadeDomain
        
    
    class NotifySpecialistOrchestrators(OneShotBehaviour):
        async def run(self):
            
            # Notify CryptoOrchestrator to Start
            print(f"{AGENT_NAME} Notifying CryptoOrchestrator Agent to start...")
            msg = Message(to=f"cryptoOrchestrator@{self.agent.spadeDomain}")
            msg.set_metadata("performative", "start_agent")
            await self.send(msg)    
            
            # Notify CryptoOrchestrator to Start
            print(f"{AGENT_NAME} Notifying NewsOrchestrator Agent to start...")
            msg = Message(to=f"newsOrchestrator@{self.agent.spadeDomain}")
            msg.set_metadata("performative", "start_agent")
            await self.send(msg)    
            
            
    class ListeningForMessages(CyclicBehaviour):
        async def run(self):
            print(f"{AGENT_NAME} Listening for results...")
        
        
    async def setup(self):
        print(f"{AGENT_NAME} Starting...")
        self.add_behaviour(self.NotifySpecialistOrchestrators())
        