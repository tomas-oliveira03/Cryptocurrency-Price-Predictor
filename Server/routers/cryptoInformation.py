from flask import jsonify, request

from db.mongoConnection import getMongoConnection

def getCryptoData(app, prefix):
    
    predictionsDB = getMongoConnection()
    
    @app.route(f"{prefix}/<cryptoCurrency>")
    def getCryptoInformation(cryptoCurrency):
        
        cryptoData = predictionsDB.find_one(
            {"coin": cryptoCurrency},
            sort=[("date", -1)]
        )
                
        if not cryptoData:
            return jsonify({"error": "CryptoCurrency not found"}), 404
        
        cryptoData["_id"] = str(cryptoData["_id"])
        return jsonify(cryptoData), 200

