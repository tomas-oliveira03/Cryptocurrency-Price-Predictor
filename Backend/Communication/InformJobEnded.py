import json

class InformJobEnded:
    
    def __init__(self, databaseCollectionName: str, providerAgentName: str = None):
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
        return json.dumps({
            "databaseCollectionName": self.databaseCollectionName,
            "providerAgentName": self.providerAgentName
        })
        