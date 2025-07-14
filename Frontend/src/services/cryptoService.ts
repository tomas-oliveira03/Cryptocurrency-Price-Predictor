import { CoinType, CryptoData, TimeRange } from '../types';

// Available coins
const availableCoins: CoinType[] = ['BTC', 'ETH', 'XRP', 'BNB', 'SOL', 'DOGE', 'TRX', 'ADA'];

// API base URL - Use a relative path instead of absolute URL to work with the proxy
const API_BASE_URL = '/api/crypto';

// Fetch crypto data from the API without fallback
export const getCryptoData = async (coin: CoinType): Promise<CryptoData> => {
  try {
    console.log(`Fetching data for ${coin} from ${API_BASE_URL}/${coin}`);
    // Remove the http://localhost:3001 prefix to use the Vite proxy
    const response = await fetch(`http://localhost:3001/api/crypto/${coin}`, {
      method: "GET",
      headers: {
        'Accept': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`Failed to fetch ${coin} data: ${response.status}`);
    }
    
    const data = await response.json();
    console.log(`Successfully fetched ${coin} data from API`);
    return data;
  } catch (error) {
    console.error(`Error fetching ${coin} data:`, error);
    throw error;
  }
};

export const filterDataByTimeRange = (data: CryptoData, timeRange: TimeRange): CryptoData => {
  if (!data.historical_price || data.historical_price.length === 0) {
    return data; // Return original data if no historical prices
  }
  
  // Parse dates safely
  const parseDates = (dateStr: string): Date => {
    try {
      // Handle various date formats that might come from the API
      if (dateStr.includes('GMT')) {
        // Format like "Mon, 29 Apr 2024 00:00:00 GMT"
        return new Date(dateStr);
      } else {
        // ISO format or simple YYYY-MM-DD
        return new Date(dateStr);
      }
    } catch (e) {
      console.warn("Invalid date format:", dateStr);
      return new Date(); // Return current date as fallback
    }
  };

  // Find latest date in the historical data
  const dates = data.historical_price.map(item => parseDates(item.date));
  const latestDate = new Date(Math.max(...dates.map(d => d.getTime())));
  
  let startDate: Date;
  switch (timeRange) {
    case '7days':
      startDate = new Date(latestDate);
      startDate.setDate(latestDate.getDate() - 7);
      break;
    case '30days':
      startDate = new Date(latestDate);
      startDate.setDate(latestDate.getDate() - 30);
      break;
    case '1year':
      startDate = new Date(latestDate);
      startDate.setDate(latestDate.getDate() - 365);
      break;
    default:
      startDate = new Date(latestDate);
      startDate.setDate(latestDate.getDate() - 30);
  }
  
  // Filter data using the startDate
  return {
    ...data,
    historical_price: data.historical_price.filter(item => {
      try {
        return parseDates(item.date) >= startDate;
      } catch (e) {
        return false;
      }
    }),
    predicted_price: data.predicted_price.filter(item => {
      try {
        return parseDates(item.date) >= startDate;
      } catch (e) {
        return false;
      }
    }),
    positive_sentiment_ratio: data.positive_sentiment_ratio.filter(item => {
      try {
        return parseDates(item.date) >= startDate;
      } catch (e) {
        return false;
      }
    })
  };
};

export const getAvailableCoins = (): CoinType[] => {
  return availableCoins;
};
