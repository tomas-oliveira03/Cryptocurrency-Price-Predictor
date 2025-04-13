import json
from spade.message import Message

async def sendMessage(agentInstance, destinationAgent, performative, payload=None):
    msg = Message(to=f"{destinationAgent}@{agentInstance.agent.spadeDomain}")
    msg.set_metadata("performative", performative)
    
    if payload:
        msg.body = json.dumps(payload)
        
    await agentInstance.send(msg)    
    