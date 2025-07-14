import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { CoinType } from '../types';
import PriceAlertPopup from './PriceAlertPopup';
import PriceAlertsList from './PriceAlertsList';

interface PriceAlertButtonProps {
  coin: CoinType;
  currentPrice: number;
}

const PriceAlertButton: React.FC<PriceAlertButtonProps> = ({ coin, currentPrice }) => {
  const [isListOpen, setIsListOpen] = useState(false);
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  // Use a ref to force refresh of alerts list when needed
  const refreshCounter = useRef(0);
  
  const handleOpenList = () => {
    setIsListOpen(true);
  };
  
  const handleCloseList = () => {
    setIsListOpen(false);
  };
  
  const handleOpenCreate = () => {
    setIsListOpen(false);
    setIsCreateOpen(true);
  };
  
  const handleCloseCreate = () => {
    setIsCreateOpen(false);
    // Go back to list after creating and refresh the list
    setIsListOpen(true);
    // Increment counter to trigger useEffect in PriceAlertsList
    refreshCounter.current += 1;
  };

  return (
    <>
      <motion.div 
        className="price-alert-button"
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        onClick={handleOpenList}
        title="Price Alerts"
      >
        <span className="bell-icon">🔔</span>
        <span className="alert-label">Price Alerts</span>
      </motion.div>
      
      <PriceAlertsList
        isOpen={isListOpen}
        onClose={handleCloseList}
        onAddNew={handleOpenCreate}
        currentCoin={coin}
        currentPrice={currentPrice}
        refreshTrigger={refreshCounter.current} // Pass refresh trigger
      />
      
      <PriceAlertPopup
        isOpen={isCreateOpen}
        onClose={handleCloseCreate}
        currentCoin={coin}
        currentPrice={currentPrice}
      />
    </>
  );
};

export default PriceAlertButton;
