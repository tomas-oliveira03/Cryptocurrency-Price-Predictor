import datetime
from flask import jsonify, request

from db.mongoConnection import getMongoConnection

def getCryptoData(app, prefix):
    
    predictionsDB, cryptoPriceDB = getMongoConnection()
    
    @app.route(f"{prefix}/<cryptoCurrency>")
    def getCryptoInformation(cryptoCurrency):
        
        cryptoData = predictionsDB.find_one(
            {"coin": cryptoCurrency},
            sort=[("date", -1)]
        )
        
        cryptoPrice = cryptoPriceDB.find_one(
            {"cryptoCurrency": cryptoCurrency},
            sort=[("date", -1)]
        )
        
                
        if not cryptoData:
            return jsonify({"error": "CryptoCurrency not found"}), 404
        
        sorted_prices = sorted(cryptoData["historical_price"], key=lambda x: x["date"])
        
        mostRecentCryptoData = sorted_prices[-1]
        currentPrice = mostRecentCryptoData["date"]
        
        print(mostRecentCryptoData)
        
        if cryptoPrice:
            print(cryptoPrice)
            
            if mostRecentCryptoData["date"] < cryptoPrice["date"]:
                currentPrice=cryptoPrice["price"]
            
        
        cryptoData["_id"] = str(cryptoData["_id"])
        cryptoData["current_price"] = currentPrice
        return jsonify(cryptoData), 200

