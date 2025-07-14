import { useState, useEffect, useCallback } from 'react';
import TimeRangeSelector from './TimeRangeSelector';
import CryptoChart from './CryptoChart';
import ModelBenchmarks from './ModelBenchmarks';
import PriceForecastCard from './PriceForecastCard';
import PriceAlertButton from './PriceAlertButton';
import PredictionBenchmarkTable from './PredictionBenchmarkTable';
import { CoinType, CryptoData, DataField, TimeRange } from '../types';
import { getCryptoData, getAvailableCoins, filterDataByTimeRange } from '../services/cryptoService';
import { websocketService } from '../services/websocketService';
import { motion } from 'framer-motion';
import { useAuth } from '../context/AuthContext';

interface CryptoDashboardProps {
  initialCoin?: CoinType;
}

const CryptoDashboard: React.FC<CryptoDashboardProps> = ({ initialCoin = 'BTC' }) => {
  const { isAuthenticated } = useAuth();
  const [selectedCoin, setSelectedCoin] = useState<CoinType>(initialCoin);
  const [selectedTimeRange, setSelectedTimeRange] = useState<TimeRange>('30days');
  const [selectedFields] = useState<DataField[]>(['historical_price', 'predicted_price', 'positive_sentiment_ratio']);
  const [cryptoData, setCryptoData] = useState<CryptoData | null>(null);
  const [originalData, setOriginalData] = useState<CryptoData | null>(null);
  const [availableCoins, setAvailableCoins] = useState<CoinType[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [priceStats, setPriceStats] = useState<{
    current: number;
    initial: number;
    changePercentage: number;
    changeAmount: number;
    nextDay: number;
    nextDayChange: number;
    nextDayChangeAmount: number;
    sevenDay: number;
    sevenDayChange: number;
    sevenDayChangeAmount: number;
  } | null>(null);
  
  // Set initial coin from props when it changes
  useEffect(() => {
    if (initialCoin !== selectedCoin) {
      setSelectedCoin(initialCoin);
    }
  }, [initialCoin]);

  const fetchCryptoData = useCallback(async (coin: CoinType) => {
    setLoading(true);
    setError(null);
    try {
      console.log(`Fetching fresh data for ${coin}...`);
      const data = await getCryptoData(coin);

      if (!data || !data.historical_price || data.historical_price.length === 0) {
        throw new Error(`No data available for ${coin}`);
      }

      console.log(`Received data for ${coin} with current price: ${data.current_price}`);
      
      // Set the original data with the current price from API
      setOriginalData(data);
      
      // Create a filtered copy that maintains the current price
      const filteredData = {
        ...filterDataByTimeRange(data, selectedTimeRange),
        current_price: data.current_price
      };
      
      // Set the filtered data with the current price preserved
      setCryptoData(filteredData);
      
      // Calculate stats with the new price data
      const stats = calculatePriceStats(filteredData);
      console.log(`Calculated stats for ${coin} - Current price: ${stats?.current}`);
      setPriceStats(stats);
    } catch (err: any) {
      console.error(`Error loading data:`, err);

      if (err.message && err.message.includes('ECONNREFUSED')) {
        setError(
          `Connection Refused: Unable to connect to the backend server at http://localhost:3001. ` +
          `Please ensure the backend server is running.`
        );
      } else if (err.message && err.message.includes('500')) {
        setError(
          `API Server Error (500): The backend server at http://localhost:3001 returned an internal error. ` +
          `Check the backend server logs for details.`
        );
      } else if (err.message && err.message.includes('Failed to fetch')) {
        setError(
          `Network Error: Failed to fetch data. ` +
          `Verify the backend server at http://localhost:3001 is running and accessible. Use console tests (testProxy) to confirm.`
        );
      } else {
        setError(`Failed to load ${coin} data: ${err.message || 'Unknown error'}`);
      }

      setCryptoData(null);
      setPriceStats(null);
    } finally {
      setLoading(false);
    }
  }, []); // Removed selectedTimeRange from dependency array

  useEffect(() => {
    setAvailableCoins(getAvailableCoins());
    fetchCryptoData(selectedCoin);
  }, [fetchCryptoData, selectedCoin]);

  useEffect(() => {
    websocketService.connect();
    
    const handlePriceUpdate = (coin: string, price: number, previousPrice: number | null) => {
      if (coin.toUpperCase() === selectedCoin) {
        console.log(`Received WebSocket update for ${coin}: ${price} (previous: ${previousPrice})`);
        
        setOriginalData(prevData => {
          if (!prevData) return null;
          
          const updatedData = {
            ...prevData,
            current_price: price
          };
          return updatedData;
        });
        
        setCryptoData(prevData => {
          if (!prevData) return null;
          
          const updatedData = {
            ...prevData,
            current_price: price
          };
          
          const newStats = calculatePriceStats(updatedData);
          setPriceStats(newStats);
          
          return updatedData;
        });
      }
    };
    
    websocketService.onPriceUpdate(handlePriceUpdate);
    
    return () => {
      websocketService.removeCallback(handlePriceUpdate);
    };
  }, [selectedCoin]);

  const calculatePriceStats = (data: CryptoData) => {
    if (!data.historical_price || !data.predicted_price || 
        data.historical_price.length === 0 || data.predicted_price.length === 0) {
      return null;
    }

    const sortedHistoricalPrices = [...data.historical_price].sort((a, b) => {
      return new Date(a.date).getTime() - new Date(b.date).getTime();
    });
    
    const sortedPredictedPrices = [...data.predicted_price].sort((a, b) => {
      return new Date(a.date).getTime() - new Date(b.date).getTime();
    });

    const initialPrice = sortedHistoricalPrices[0].price;
    const currentPrice = data.current_price;
    const changePercentage = ((currentPrice - initialPrice) / initialPrice) * 100;
    const changeAmount = currentPrice - initialPrice;
    const nextDayPrice = sortedPredictedPrices[0].price;
    const nextDayChange = ((nextDayPrice - currentPrice) / currentPrice) * 100;
    const nextDayChangeAmount = nextDayPrice - currentPrice;
    const sevenDayIndex = Math.min(6, sortedPredictedPrices.length - 1);
    const sevenDayPrice = sortedPredictedPrices[sevenDayIndex].price;
    const sevenDayChange = ((sevenDayPrice - currentPrice) / currentPrice) * 100;
    const sevenDayChangeAmount = sevenDayPrice - currentPrice;
    
    return {
      current: currentPrice,
      initial: initialPrice,
      changePercentage,
      changeAmount,
      nextDay: nextDayPrice,
      nextDayChange,
      nextDayChangeAmount,
      sevenDay: sevenDayPrice,
      sevenDayChange,
      sevenDayChangeAmount
    };
  };

  const handleCoinChange = (coin: CoinType) => {
    console.log(`Changing coin to: ${coin}`);
    
    // Clear previous data when changing coins to prevent stale data display
    setCryptoData(null);
    setPriceStats(null);
    
    // Set the new coin (URL update happens in the useEffect)
    setSelectedCoin(coin);
    
    // Always force a fresh data fetch when changing coins
    fetchCryptoData(coin);
  };

  const handleTimeRangeChange = (timeRange: TimeRange) => {
    setSelectedTimeRange(timeRange);
    if (originalData) {
      // Ensure current_price is maintained when filtering data for time range changes
      const filteredData = {
        ...filterDataByTimeRange(originalData, timeRange),
        current_price: originalData.current_price // Preserve the current price
      };
      setCryptoData(filteredData);
      
      const stats = calculatePriceStats(filteredData);
      setPriceStats(stats);
    }
  };

  const formatCalculationDate = (dateString?: string) => {
    if (!dateString) return '';

    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading && !cryptoData) {
    return (
      <div className="loading-indicator">
        <p>Loading cryptocurrency data...</p>
        <p className="loading-subtext">Connecting to API server at http://localhost:3001</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-message">
        <h2>Error Loading Cryptocurrency Data</h2>
        <p>{error}</p>
        <p>
          <strong>Troubleshooting steps:</strong>
        </p>
        <ol>
          <li><strong>Verify the backend API server is running at http://localhost:3001.</strong> (Check its console!)</li>
          <li>Check the backend server's console output for any errors.</li>
          <li>Restart both backend and frontend servers after making changes.</li>
        </ol>
        <div className="error-actions">
          <button 
            onClick={() => fetchCryptoData(selectedCoin)} 
            className="retry-button"
          >
            Retry
          </button>
        </div>
        <div className="api-test">
          <p>Run these commands in the browser console to diagnose:</p>
          <pre>window.testApiConnection.testDirect()  // Checks backend reachability</pre>
          <pre>window.testApiConnection.testProxy()   // Checks if proxy works</pre>
        </div>
      </div>
    );
  }

  if (!cryptoData || !priceStats) {
    return <div className="no-data-message">No data available for {selectedCoin}</div>;
  }

  return (
    <div className="crypto-dashboard">
      <h1>Cryptocurrency Dashboard</h1>
      
      {/* New Crypto Header - displays name and image */}
      <motion.div 
        className="crypto-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <img 
          src={`/crypto-logos/${selectedCoin}.png`} 
          alt={`${selectedCoin} logo`} 
          className="crypto-header-image" 
          onError={(e) => {
            // Fallback if image fails to load
            (e.target as HTMLImageElement).src = "/crypto-logos/generic-crypto.png";
          }}
        />
        <div className="crypto-header-content">
          <h2 className="crypto-header-name">
            {selectedCoin === 'BTC' ? 'Bitcoin' : 
             selectedCoin === 'ETH' ? 'Ethereum' :
             selectedCoin === 'XRP' ? 'Ripple' :
             selectedCoin === 'BNB' ? 'Binance Coin' :
             selectedCoin === 'SOL' ? 'Solana' :
             selectedCoin === 'DOGE' ? 'Dogecoin' :
             selectedCoin === 'TRX' ? 'TRON' :
             selectedCoin === 'ADA' ? 'Cardano' : selectedCoin}
          </h2>
          <span className="crypto-header-symbol">{selectedCoin}</span>
        </div>
      </motion.div>
      
      <div className="current-price-container">
        {priceStats && (
          <PriceForecastCard
            key={`${selectedCoin}-current-${priceStats.current}`} // Force re-render on coin or price change
            title="Current Price"
            price={priceStats.current}
            changePercentage={priceStats.changePercentage}
            changeAmount={priceStats.changeAmount}
            coin={selectedCoin}
          />
        )}
      </div>

      <div className="price-alert-container">
        {priceStats && (
          <>
            {isAuthenticated ? (
              <PriceAlertButton 
                coin={selectedCoin} 
                currentPrice={priceStats.current} 
              />
            ) : (
              <motion.div 
                className="price-alert-button disabled"
                whileHover={{ scale: 1.05 }}
                title="Login or register to set price alerts"
              >
                <span className="bell-icon">🔔</span>
                <span className="alert-label">Price Alerts</span>
                <div className="auth-tooltip">
                  Login or register to set price alerts
                </div>
              </motion.div>
            )}
          </>
        )}
      </div>
      
      {cryptoData ? (
        <div className="chart-with-controls">
          <div className="chart-time-selector compact">
            <TimeRangeSelector 
              selectedTimeRange={selectedTimeRange}
              onChange={handleTimeRangeChange}
            />
          </div>
          <CryptoChart data={cryptoData} selectedFields={selectedFields} />
        </div>
      ) : (
        <div className="no-data-message">
          {loading ? "Loading data..." : "No data available"}
        </div>
      )}
      
      <div className="forecast-cards-container">
        {priceStats && (
          <>
            <PriceForecastCard
              key={`${selectedCoin}-nextday-${priceStats.nextDay}`}
              title="Next Day Forecast"
              price={priceStats.nextDay}
              changePercentage={priceStats.nextDayChange}
              changeAmount={priceStats.nextDayChangeAmount}
              coin={selectedCoin}
            />
            <PriceForecastCard
              key={`${selectedCoin}-sevenday-${priceStats.sevenDay}`}
              title="7-Day Forecast"
              price={priceStats.sevenDay}
              changePercentage={priceStats.sevenDayChange}
              changeAmount={priceStats.sevenDayChangeAmount}
              coin={selectedCoin}
            />
          </>
        )}
      </div>

      {cryptoData.prediction_benchmarks && cryptoData.prediction_benchmarks.length > 0 && (
        <PredictionBenchmarkTable benchmarks={cryptoData.prediction_benchmarks} />
      )}
      
      {cryptoData.model_benchmarks && (
        <ModelBenchmarks benchmarks={cryptoData.model_benchmarks} />
      )}
      
      {cryptoData.date && (
        <div className="calculation-date">
          <p>Prediction calculated on: <strong>{formatCalculationDate(cryptoData.date)}</strong></p>
        </div>
      )}
    </div>
  );
};

export default CryptoDashboard;
