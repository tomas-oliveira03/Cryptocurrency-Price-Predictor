import { motion, AnimatePresence } from 'framer-motion';
import { useEffect, useState, useRef } from 'react';
import websocketService from '../services/websocketService';

interface PriceForecastCardProps {
  title: string;
  price: number;
  changePercentage: number;
  changeAmount?: number;
  subtitle?: string;
  coin?: string; // Add coin prop to know which updates to listen for
}

const PriceForecastCard: React.FC<PriceForecastCardProps> = ({
  title,
  price,
  changePercentage,
  changeAmount,
  subtitle,
  coin
}) => {
  // Internal state to hold the displayed values, potentially updated by WebSocket
  const [currentPrice, setCurrentPrice] = useState<number>(price);
  const [currentChangePercentage, setCurrentChangePercentage] = useState<number>(changePercentage);
  const [currentChangeAmount, setCurrentChangeAmount] = useState<number | undefined>(changeAmount);
  const [isUpdating, setIsUpdating] = useState<boolean>(false);

  // Ref to store the initial reference price for calculating changes in the "Current" card (used by WS)
  const referencePriceRef = useRef<number | null>(null);
  // Ref to store the fixed predicted price for forecast cards
  const baselinePriceRef = useRef<number>(price);
  // Ref to track if the component has mounted and initial props are set
  const isMountedRef = useRef(false);

  // --- Initialize State and References / Handle Prop Updates ---
  useEffect(() => {
    // Logic for Forecast cards OR initial mount of ANY card
    if (title.includes('Forecast') || !isMountedRef.current) {
      setCurrentPrice(price);
      setCurrentChangePercentage(changePercentage);
      setCurrentChangeAmount(changeAmount);
      baselinePriceRef.current = price; // Store the initial price (current or predicted)

      // Calculate the reference price for the "Current" card only on first setup
      if (title.includes('Current') && !isMountedRef.current) {
        if (changeAmount !== undefined) {
          referencePriceRef.current = price - changeAmount;
        } else if (changePercentage !== 0 && price !== 0) { // Avoid division by zero
          referencePriceRef.current = price / (1 + (changePercentage / 100));
        } else {
          referencePriceRef.current = price;
        }
        console.log(`Current Price Card Initialized for ${coin}. Initial Price: ${price}, Reference Price: ${referencePriceRef.current}`);
      }

      if (!isMountedRef.current) {
        isMountedRef.current = true;
      }
      console.log(`State updated from props for ${title} (${coin}) - Initial Mount or Forecast Card.`);

    } else if (title.includes('Current') && isMountedRef.current) {
      // Logic for "Current" card AFTER initial mount (props changed due to time range)
      // Keep the WebSocket-driven price (currentPrice state), but update the change % and amount from props
      // These props reflect the change over the *newly selected time range*.
      setCurrentChangePercentage(changePercentage);
      setCurrentChangeAmount(changeAmount);
      console.log(`Props updated for Current Price Card (${coin}): Change %/Amount updated from props, Price maintained.`);
    }

    // Dependency array: Rerun if the fundamental identity (coin, title) or the data values change.
  }, [coin, title, price, changePercentage, changeAmount]);


  // --- WebSocket Update Logic ---
  useEffect(() => {
    if (coin) {
      websocketService.connect();

      const handlePriceUpdate = (updatedCoin: string, newPrice: number, _previousPrice: number | null) => {
        if (updatedCoin === coin) {
          if (title.includes('Current')) {
            // --- Update Current Price Card ---
            if (referencePriceRef.current === null) {
              console.warn(`Reference price not set for ${coin}, cannot calculate accurate change.`);
              return; // Don't update if reference isn't ready
            }
            console.log(`WS: Updating Current price for ${coin}: $${newPrice}`);
            setCurrentPrice(newPrice); // Update displayed price

            // Recalculate changes based on the *original reference price*
            const newChangeAmount = newPrice - referencePriceRef.current;
            const newChangePercentage = referencePriceRef.current === 0 ? 0 : (newChangeAmount / referencePriceRef.current) * 100;

            setCurrentChangePercentage(newChangePercentage);
            setCurrentChangeAmount(newChangeAmount);
            console.log(`WS: Current metrics updated: ${newChangePercentage.toFixed(2)}% ($${newChangeAmount.toFixed(2)})`);

          } else if (title.includes('Forecast')) {
            // --- Update Forecast Card ---
            const predictedPrice = baselinePriceRef.current; // Use the fixed predicted price
            console.log(`WS: Recalculating Forecast for ${coin} based on new current price $${newPrice}. Predicted: $${predictedPrice}`);

            // Recalculate change based on the *new current price* and the *fixed predicted price*
            const newChangeAmount = predictedPrice - newPrice;
            const newChangePercentage = newPrice === 0 ? 0 : (newChangeAmount / newPrice) * 100;

            setCurrentChangePercentage(newChangePercentage);
            setCurrentChangeAmount(newChangeAmount);
            console.log(`WS: Forecast metrics updated: ${newChangePercentage.toFixed(2)}% ($${newChangeAmount.toFixed(2)})`);
          }

          // Trigger visual update effect
          setIsUpdating(true);
          setTimeout(() => setIsUpdating(false), 1500); // Shorter highlight
        }
      };

      websocketService.onPriceUpdate(handlePriceUpdate);

      return () => {
        websocketService.removeCallback(handlePriceUpdate);
      };
    }
  }, [coin, title]); // Rerun if coin or title changes

  // --- Formatting ---
  const formattedPrice = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(currentPrice);

  const formattedChangeAmount = currentChangeAmount !== undefined
    ? new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      }).format(Math.abs(currentChangeAmount))
    : null;

  const isPositive = currentChangePercentage >= 0;
  const changeColor = isPositive ? '#4ade80' : '#f87171'; // Tailwind green-500 / red-500
  const changeSymbol = isPositive ? '+' : '';

  // --- Render ---
  return (
    <motion.div
      className="price-forecast-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="card-title">{title}</div>

      {/* Price Value */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`price-${currentPrice}`} // Key ensures animation on price change
          className="price-value"
          initial={{ opacity: 0.7, y: -5 }}
          animate={{
            opacity: 1,
            y: 0,
            backgroundColor: isUpdating ? 'rgba(79, 172, 254, 0.1)' : 'transparent', // Highlight on update
            transition: { duration: 0.3 }
          }}
          exit={{ opacity: 0, y: 5, transition: { duration: 0.2 } }}
        >
          {formattedPrice}
        </motion.div>
      </AnimatePresence>

      {/* Price Change */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`change-${currentChangePercentage.toFixed(2)}-${currentChangeAmount?.toFixed(2)}`} // Key ensures animation on change update
          className="price-change"
          style={{ color: changeColor }}
          initial={{ opacity: 0.7, y: -5 }}
          animate={{ opacity: 1, y: 0, transition: { duration: 0.3, delay: 0.05 } }}
          exit={{ opacity: 0, y: 5, transition: { duration: 0.2 } }}
        >
          {changeSymbol}{currentChangePercentage.toFixed(2)}%
          {formattedChangeAmount && (
            <span className="price-change-amount"> ({changeSymbol}{formattedChangeAmount})</span>
          )}
        </motion.div>
      </AnimatePresence>

      {subtitle && <div className="price-subtitle">{subtitle}</div>}
    </motion.div>
  );
};

export default PriceForecastCard;
