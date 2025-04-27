import os
from bson import CodecOptions
from pymongo import MongoClient, UpdateOne

class Notifications:
    
    def __init__(self, SHOW_LOGS=True):
        self.SHOW_LOGS=SHOW_LOGS
        mongoDBURI = os.getenv("MONGODB_URI")
        
        if not mongoDBURI:
            raise ValueError("Please set the mongoDB environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.usersDB = mongoClient['ASM-Users'].get_collection('users', codec_options=CodecOptions(tz_aware=True))
        self.notificationsDB = mongoClient['ASM-Users'].get_collection('notifications', codec_options=CodecOptions(tz_aware=True))


    def checkNewPossibleNotifications(self, cryptoData):
        # Step 1: Query active notifications matching the cryptoCurrency and monitoredPriceType
        notificationsCursor = self.notificationsDB.find({
            "isActive": True,
            "cryptoCurrency": cryptoData["coin"],
            "monitoredPriceType": cryptoData["monitoredPriceType"]
        })

        aboveNotifications = []
        belowNotifications = []

        # Step 2: Process each notification
        for notification in notificationsCursor:
            alertCondition = notification.get("alertCondition")
            targetPrice = notification.get("price")
            currentPrice = cryptoData["price"]
            userId = notification.get("userId")
            
            # Find the respective user
            user = self.usersDB.find_one({"_id": userId})
            if not user:
                continue
            
            # Attach the user's email into the notification
            notification["email"] = user.get("email")

            if alertCondition == "ABOVE" and currentPrice > targetPrice:
                aboveNotifications.append(notification)
                
            elif alertCondition == "BELOW" and currentPrice < targetPrice:
                belowNotifications.append(notification)

        if self.SHOW_LOGS:
            print(f"Above Notifications to trigger: {aboveNotifications}")
            print(f"Below Notifications to trigger: {belowNotifications}")

        return aboveNotifications, belowNotifications
            
   
    
        
if __name__ == "__main__":
    notifications = Notifications()
    
    cryptoData = {
        "coin": "BTC",
        "price": 2,
        "monitoredPriceType": "REAL"
    }
    
    notifications.checkNewPossibleNotifications(cryptoData)
