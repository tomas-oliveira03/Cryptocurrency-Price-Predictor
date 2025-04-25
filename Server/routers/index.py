from flask import request
from datetime import datetime
from utils.logger import logger  
from routers.cryptoInformation import getCryptoData
from routers.auth import authentication 
from routers.notification import notification
from db.mongoConnection import getMongoConnectionForCrypto
from db.mongoConnection import getMongoConnectionForUser

def registerRoutes(app, path, wsManager):
    # Optional: Request timing
    @app.before_request
    def before_request():
        request.start_time = datetime.now()

    @app.after_request
    def after_request(response):
        if hasattr(request, 'start_time'):
            duration = datetime.now() - request.start_time
            logger.info(f"Done in {duration.total_seconds():.2f}s - {response.status_code}")
        return response


    predictionsDB, cryptoPriceDB = getMongoConnectionForCrypto()
    userDB, notificationDB = getMongoConnectionForUser()

    # Register specialist agent routes
    getCryptoData(app, f"{path}/crypto/", wsManager, predictionsDB, cryptoPriceDB)
    authentication(app, f"{path}/auth/", userDB)
    notification(app, f"{path}/notification/", userDB, notificationDB)
