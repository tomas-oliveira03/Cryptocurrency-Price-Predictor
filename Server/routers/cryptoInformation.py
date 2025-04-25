from flask import json, jsonify, request

def getCryptoData(app, prefix, wsManager, predictionsDB, cryptoPriceDB):
    
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


    @app.route(f"{prefix}/update", methods=["POST"])
    def updateCryptoInformation():
        try:
            # Try reading the raw data and manually parsing it
            raw_data = request.data
            data = json.loads(raw_data)
        except Exception as e:
            return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

        if not isinstance(data, list):
            return jsonify({"error": "Invalid data format: expected a list of coin info objects"}), 400

        for item in data:
            if not isinstance(item, dict) or "coin" not in item or "price" not in item:
                return jsonify({"error": "Invalid item format: each item must have 'coin' and 'price' keys"}), 400

        wsManager.broadcast(data)
        return jsonify({"message": "Broadcast sent"}), 200