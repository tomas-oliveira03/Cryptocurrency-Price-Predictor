import os
from pymongo import MongoClient
from bson import CodecOptions

def getMongoConnectionForCrypto():
    mongoDBURI = os.getenv("MONGODB_URI")
    if not mongoDBURI:
        raise ValueError("Please set the MONGODB_URI environment variable first.")

    # Database connection
    mongoClient = MongoClient(mongoDBURI)
    
    # Get commonly used collections
    predictionsDB = mongoClient['ASM'].get_collection('predictions', codec_options=CodecOptions(tz_aware=True))
    cryptoPriceDB = mongoClient['ASM'].get_collection('crypto-price', codec_options=CodecOptions(tz_aware=True))
    
    return predictionsDB, cryptoPriceDB


def getMongoConnectionForUser():
    mongoDBURI = os.getenv("MONGODB_URI")
    if not mongoDBURI:
        raise ValueError("Please set the MONGODB_URI environment variable first.")

    # Database connection
    mongoClient = MongoClient(mongoDBURI)
    
    # Get commonly used collections
    userDB = mongoClient['ASM-Users'].get_collection('users', codec_options=CodecOptions(tz_aware=True))
    
    return userDB