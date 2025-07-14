import { CoinType } from '../types';
import { motion } from 'framer-motion';
import { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';

interface CoinSelectorProps {
  selectedCoin: CoinType;
  availableCoins: CoinType[];
  onChange: (coin: CoinType) => void;
}

// Get cryptocurrency logo path
const getCoinLogoPath = (coin: CoinType): string => {
  return `/crypto-logos/${coin}.png`;
};

const CoinSelector: React.FC<CoinSelectorProps> = ({ selectedCoin, availableCoins, onChange }) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLDivElement>(null);
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, left: 0, width: 0 });

  // Calculate dropdown position based on button position
  useEffect(() => {
    if (buttonRef.current && dropdownOpen) {
      const rect = buttonRef.current.getBoundingClientRect();
      setDropdownPosition({
        top: rect.bottom + window.scrollY,
        left: rect.left + window.scrollX,
        width: rect.width
      });
    }
  }, [dropdownOpen]);

  // Prevent page scrolling when scrolling inside dropdown
  useEffect(() => {
    if (!dropdownOpen || !dropdownRef.current) return;

    const dropdown = dropdownRef.current;
    
    const handleWheel = (e: WheelEvent) => {
      // Prevent default only if we're scrolling inside the dropdown
      const isScrollingDown = e.deltaY > 0;
      const isScrollingUp = e.deltaY < 0;
      const isAtTop = dropdown.scrollTop === 0;
      const isAtBottom = dropdown.scrollHeight - dropdown.clientHeight - dropdown.scrollTop <= 1;
      
      // Only prevent default when trying to scroll beyond bounds
      if ((isScrollingUp && isAtTop) || (isScrollingDown && isAtBottom)) {
        e.preventDefault();
      }
    };
    
    // Add passive: false to ensure we can preventDefault()
    dropdown.addEventListener('wheel', handleWheel, { passive: false });
    
    return () => {
      dropdown.removeEventListener('wheel', handleWheel);
    };
  }, [dropdownOpen]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node) &&
          buttonRef.current && !buttonRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  return (
    <motion.div 
      className="coin-selector"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <label htmlFor="coin-select">
      </label>
      
      {/* Custom dropdown selector */}
      <div className="custom-select-container">
        {/* Selected coin display - acts as dropdown toggle */}
        <div 
          ref={buttonRef}
          className={`selected-coin-display ${dropdownOpen ? 'active' : ''}`}
          onClick={() => setDropdownOpen(!dropdownOpen)}
        >
          <img 
            src={getCoinLogoPath(selectedCoin)} 
            alt={`${selectedCoin} logo`} 
            className="coin-logo"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none';
              (e.target as HTMLImageElement).nextSibling!.style.display = 'inline';
            }}
          />
          <span className="coin-icon-fallback" style={{ display: 'none' }}>
            {selectedCoin}
          </span>
          <span className="selected-coin-text">{selectedCoin}</span>
          <span className="dropdown-arrow">▼</span>
        </div>
        
        {/* Dropdown options - Rendered with Portal to avoid being clipped */}
        {dropdownOpen && createPortal(
          <motion.div 
            ref={dropdownRef}
            className="coin-dropdown"
            style={{
              position: 'absolute',
              top: `${dropdownPosition.top}px`,
              left: `${dropdownPosition.left}px`,
              width: `${dropdownPosition.width}px`
            }}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2 }}
          >
            {availableCoins.map((coin) => (
              <div 
                key={coin} 
                className={`coin-option ${selectedCoin === coin ? 'selected' : ''}`}
                onClick={() => {
                  onChange(coin);
                  setDropdownOpen(false);
                }}
              >
                <img 
                  src={getCoinLogoPath(coin)} 
                  alt={`${coin} logo`} 
                  className="coin-logo"
                  onError={(e) => {
                    (e.target as HTMLImageElement).style.display = 'none';
                    (e.target as HTMLImageElement).nextSibling!.style.display = 'inline';
                  }}
                />
                <span className="coin-icon-fallback" style={{ display: 'none' }}>
                  {coin}
                </span>
                <span className="coin-option-text">{coin}</span>
              </div>
            ))}
          </motion.div>,
          document.body
        )}
      </div>
      
      {/* Hidden actual select for form submission if needed */}
      <select 
        id="coin-select" 
        value={selectedCoin} 
        onChange={(e) => onChange(e.target.value as CoinType)}
        className="hidden-select"
        aria-hidden="true"
      >
        {availableCoins.map((coin) => (
          <option key={coin} value={coin}>{coin}</option>
        ))}
      </select>
    </motion.div>
  );
};

export default CoinSelector;
