import { useEffect, useState } from 'react';
import { useParams, Navigate } from 'react-router-dom';
import CryptoDashboard from '../components/CryptoDashboard';
import { CoinType } from '../types';
import { getAvailableCoins } from '../services/cryptoService';

const CoinPage = () => {
  const { coinSymbol } = useParams<{ coinSymbol: string }>();
  const [isValidCoin, setIsValidCoin] = useState<boolean | null>(null);
  const availableCoins = getAvailableCoins();

  useEffect(() => {
    // Check if the provided coin symbol is valid
    if (coinSymbol) {
      const isValid = availableCoins.includes(coinSymbol.toUpperCase() as CoinType);
      setIsValidCoin(isValid);
    } else {
      setIsValidCoin(false);
    }
  }, [coinSymbol, availableCoins]);

  // If still checking validity, show nothing (prevents flicker)
  if (isValidCoin === null) {
    return null;
  }

  // If invalid coin symbol, redirect to BTC
  if (!isValidCoin) {
    return <Navigate to="/BTC" replace />;
  }

  // Convert to proper coin type and render dashboard with the specified coin
  const normalizedCoin = coinSymbol?.toUpperCase() as CoinType;
  return <CryptoDashboard initialCoin={normalizedCoin} />;
};

export default CoinPage;
