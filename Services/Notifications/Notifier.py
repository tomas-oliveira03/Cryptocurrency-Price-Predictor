import os
import sys
from bson import CodecOptions
from dotenv import load_dotenv
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Notifications.mailer import sendCryptoEmail

class Notifications:
    
    def __init__(self, SHOW_LOGS=True):
        load_dotenv()
        self.SHOW_LOGS=SHOW_LOGS
        mongoDBURI = os.getenv("MONGODB_URI")
        
        self.senderEmail = os.getenv("GMAIL_USERNAME")
        self.senderPassword = os.getenv("GMAIL_PASSWORD_SMTP")
        self.smtpServer = os.getenv("SMTP_SERVER")
        self.smtpPort = int(os.getenv("SMTP_PORT"))
        if not mongoDBURI or not self.senderEmail or not self.senderPassword or not self.smtpServer or not self.smtpPort:
            raise ValueError("Please set the environment variable first.")

        # Database connection
        mongoClient = MongoClient(mongoDBURI)
        self.usersDB = mongoClient['ASM-Users'].get_collection('users', codec_options=CodecOptions(tz_aware=True))
        self.notificationsDB = mongoClient['ASM-Users'].get_collection('notifications', codec_options=CodecOptions(tz_aware=True))

    def checkNewPossibleNotificationsForAllCoins(self, allCoinsData):
        for coinData in allCoinsData:
            self.checkNewPossibleNotificationsForCoin(coinData)


    def checkNewPossibleNotificationsForCoin(self, cryptoData):
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


        self.notifyUsers(aboveNotifications, belowNotifications, cryptoData)
            
   
    def notifyUsers(self, aboveNotifications, belowNotifications, cryptoData):
        currentPrice = cryptoData["price"]
        coinName = cryptoData["coin"]
        monitoredPriceType = cryptoData["monitoredPriceType"]

        for notification in aboveNotifications + belowNotifications:
            userEmail = notification.get("email")
            targetPrice = notification.get("price")
            alertCondition = notification.get("alertCondition")
            
            # These are still calculated but won't be displayed in the email
            priceDifferential = currentPrice - targetPrice
            percentageChange = (priceDifferential / targetPrice) * 100

            if self.SHOW_LOGS:
                print(f"Sending notification to {userEmail} for {coinName}")

            sendCryptoEmail(
                SHOW_LOGS=self.SHOW_LOGS,
                senderEmail=self.senderEmail,
                senderPassword=self.senderPassword,
                smtpServer=self.smtpServer,
                smtpPort=self.smtpPort,
                coinName=coinName,
                currentPrice=currentPrice,
                percentageChange=percentageChange,
                priceDifferential=priceDifferential,
                recipientEmail=userEmail,
                alertCondition=alertCondition,
                targetPrice=targetPrice,
                monitoredPriceType=monitoredPriceType
            )


if __name__ == "__main__":
    notifications = Notifications()
    
    cryptoData = [{
        "coin": "BTC",
        "price": 2,
        "monitoredPriceType": "PREDICTED"
    }]
    
    notifications.checkNewPossibleNotificationsForAllCoins(cryptoData)
