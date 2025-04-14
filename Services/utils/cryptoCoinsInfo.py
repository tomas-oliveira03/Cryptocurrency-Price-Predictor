def getTopCoins():
    return ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'DOGE', 'TRX', 'ADA']

def getTickerToFullNameMap():
    return {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'XRP': 'ripple',
        'BNB': 'binancecoin',
        'SOL': 'solana',
        'DOGE': 'dogecoin',
        'TRX': 'tron',
        'ADA': 'cardano'
    }
    
def getFullNameToTickerMap():
    ticker_to_id = getTickerToFullNameMap()
    return {v: k for k, v in ticker_to_id.items()}