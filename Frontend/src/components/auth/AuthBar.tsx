import { Link, useParams } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { motion } from 'framer-motion';
import { CoinType } from '../../types';
import CoinSelector from '../CoinSelector';
import { useNavigate } from 'react-router-dom';
import { getAvailableCoins } from '../../services/cryptoService';
import { useState, useEffect } from 'react';

const AuthBar = () => {
  const { isAuthenticated, user, logout } = useAuth();
  const { coinSymbol } = useParams<{ coinSymbol: string }>();
  const navigate = useNavigate();
  const [availableCoins, setAvailableCoins] = useState<CoinType[]>([]);
  
  useEffect(() => {
    setAvailableCoins(getAvailableCoins());
  }, []);
  
  // Handle coin change from the selector
  const handleCoinChange = (coin: CoinType) => {
    navigate(`/${coin}`);
  };
  
  const selectedCoin = (coinSymbol?.toUpperCase() || 'BTC') as CoinType;

  return (
    <motion.div 
      className="auth-bar"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div className="auth-container">
        {/* Coin selector in the auth bar */}
        <div className="auth-coin-selector">
          <CoinSelector
            selectedCoin={selectedCoin}
            availableCoins={availableCoins}
            onChange={handleCoinChange}
          />
        </div>
        
        {isAuthenticated ? (
          <div className="user-info">
            <span className="welcome-text">Welcome, {user?.name || 'User'}</span>
            <motion.button 
              onClick={logout}
              className="auth-button logout-button"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Logout
            </motion.button>
          </div>
        ) : (
          <div className="auth-buttons">
            <Link to="/login">
              <motion.button 
                className="auth-button login-button"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Login
              </motion.button>
            </Link>
            <Link to="/register">
              <motion.button 
                className="auth-button register-button"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Register
              </motion.button>
            </Link>
          </div>
        )}
      </div>
    </motion.div>
  );
};

export default AuthBar;
