import json

class InformJobEnded:
    
    def __init__(self, agentJid: str, databaseCollectionName: str, providerAgentName: str = None):
        self.agentJid = agentJid
        self.databaseCollectionName = databaseCollectionName
        self.providerAgentName = providerAgentName
        
        
    def getDatabaseCollectionName(self):
        return self.databaseCollectionName
    
    def getProviderAgentName(self):
        return self.providerAgentName
    
    
    def setDatabaseCollectionName(self, databaseCollectionName:str):
        self.databaseCollectionName = databaseCollectionName
        
    def setProviderAgentName(self, providerAgentName:str):
        self.providerAgentName = providerAgentName
        
    
    def toString(self):
        return "InformJobEnded: " + json.dumps({
            "agentJid": self.agentJid,
            "databaseCollectionName": self.databaseCollectionName,
            "providerAgentName": self.providerAgentName
        })
        