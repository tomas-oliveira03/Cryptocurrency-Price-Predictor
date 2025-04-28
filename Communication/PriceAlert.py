import json

class PriceAlert:
    
    def __init__(self, monitoredPriceType: str = None):
        self.monitoredPriceType = monitoredPriceType
        self.allCryptoPrices = []
        
        
    def getAllCryptoPrices(self):
        return self.allCryptoPrices
    
    
    def addCryptoPrice(self, coin:str, price:int):
        self.allCryptoPrices.append(
            {
                "coin": coin,
                "price": price,
                "monitoredPriceType": self.monitoredPriceType
            }
        )
        
        
    def setAllCryptoPrices(self, allCryptoPrices: list):
        updatedPrices = []
        for price in allCryptoPrices:
            updatedPrice = {**price, "monitoredPriceType": self.monitoredPriceType}
            updatedPrices.append(updatedPrice)
        
        self.allCryptoPrices = updatedPrices
    
    
    def clearCryptoPrices(self):
        self.allCryptoPrices = []
    
    
    def toString(self):
        return json.dumps(self.allCryptoPrices, indent=4)
        