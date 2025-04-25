from datetime import datetime, timezone, timedelta
from pymongo import DESCENDING

def coinBenchmark(cryptoCoin, predictionsDB, cryptoPriceDB):
    results = []
    now = datetime.now(timezone.utc)

    for i in range(7):
        predictionDay = now - timedelta(days=i + 1)
        realDay = predictionDay + timedelta(days=1)

        # Skip yesterday's prediction (i.e., don't fetch prediction for yesterday)
        if predictionDay.date() == (now - timedelta(days=1)).date():
            continue
        
        # Strip time from the prediction day date (only use date part for comparison)
        predStart = predictionDay.replace(hour=0, minute=0, second=0, microsecond=0)
        predEnd = predictionDay.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Fetch the prediction data, strip time, and compare only the date
        predictionDoc = predictionsDB.find_one(
            {
                "date": {"$gte": predStart, "$lte": predEnd},
                "coin": cryptoCoin
            },
            sort=[("date", DESCENDING)]
        )

        # Strip time from real price day date for comparison
        realStart = realDay.replace(hour=0, minute=0, second=0, microsecond=0)
        realEnd = realDay.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Fetch the real price data, also strip time
        priceDoc = cryptoPriceDB.find_one(
            {
                "date": {"$gte": realStart, "$lte": realEnd},
                "cryptoCurrency": cryptoCoin
            },
            sort=[("date", DESCENDING)]
        )

        if predictionDoc and priceDoc:
            # Fetch the prediction and real price values
            predicted_entry = predictionDoc.get("predicted_price", [])
            if predicted_entry:
                results.append({
                    "predictedPrice": predicted_entry[0]["price"],
                    "realPrice": priceDoc["price"],
                    "benchmarkDate": predicted_entry[0]["date"]
                })

    return results
