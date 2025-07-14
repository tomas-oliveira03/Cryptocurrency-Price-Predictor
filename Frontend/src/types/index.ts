export interface PriceDataPoint {
  date: string;
  price: number;
}

export interface SentimentDataPoint {
  date: string;
  sentiment: number;
}

export interface ModelBenchmarks {
  mae: number;
  mape: number;
  mse: number;
  r2: number;
  rmse: number;
}

export interface CoinPriceUpdate {
  coin: string;
  price: number;
}

export interface PredictionBenchmark {
  benchmarkDate: string;
  predictedPrice: number;
  realPrice: number;
}

export interface CryptoData {
  _id?: string;
  coin: string;
  date?: string; // Date when calculation was made
  current_price: number; 
  historical_price: PriceDataPoint[];
  predicted_price: PriceDataPoint[];
  positive_sentiment_ratio: SentimentDataPoint[];
  model_benchmarks?: ModelBenchmarks;
  prediction_benchmarks?: PredictionBenchmark[]; // New field for previous predictions
}

export type TimeRange = '7days' | '30days' | '1year';

export type CoinType = 'BTC' | 'ETH' | 'XRP' | 'BNB' | 'SOL' | 'DOGE' | 'TRX' | 'ADA';

export type DataField = 'historical_price' | 'predicted_price' | 'positive_sentiment_ratio';

// New types for price alerts
export type AlertType = 'real-time' | 'predicted';

export type AlertCondition = 'above' | 'below';

export interface PriceAlert {
  id: string;
  coin: CoinType;
  type: AlertType;
  condition: AlertCondition;
  threshold: number;
  active: boolean;
  createdAt: string;
}
