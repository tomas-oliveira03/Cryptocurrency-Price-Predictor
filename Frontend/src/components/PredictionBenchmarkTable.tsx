import { PredictionBenchmark } from '../types';
import { motion } from 'framer-motion';

interface PredictionBenchmarkTableProps {
  benchmarks: PredictionBenchmark[];
}

const PredictionBenchmarkTable: React.FC<PredictionBenchmarkTableProps> = ({ benchmarks }) => {
  // Function to format date strings
  const formatDate = (dateString: string): string => {
    const options: Intl.DateTimeFormatOptions = { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  // Function to calculate difference and percentage
  const calculateDiff = (real: number, predicted: number) => {
    const diff = real - predicted;
    const percentage = (diff / predicted) * 100;
    return { diff, percentage };
  };

  return (
    <motion.div 
      className="prediction-benchmark-table"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3>Previous Predicted Prices</h3>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Predicted Price ($)</th>
              <th>Actual Price ($)</th>
              <th>Difference ($)</th>
              <th>Difference (%)</th>
            </tr>
          </thead>
          <tbody>
            {benchmarks.map((benchmark, index) => {
              const { diff, percentage } = calculateDiff(
                benchmark.realPrice, 
                benchmark.predictedPrice
              );
              
              const isPositive = diff >= 0;
              const diffColor = isPositive ? '#4ade80' : '#f87171'; // green or red
              
              return (
                <motion.tr 
                  key={benchmark.benchmarkDate}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <td>{formatDate(benchmark.benchmarkDate)}</td>
                  <td>${benchmark.predictedPrice.toLocaleString(undefined, { 
                    maximumFractionDigits: 2 
                  })}</td>
                  <td>${benchmark.realPrice.toLocaleString(undefined, { 
                    maximumFractionDigits: 2 
                  })}</td>
                  <td style={{ color: diffColor }}>
                    {isPositive ? '+' : ''}
                    ${Math.abs(diff).toLocaleString(undefined, { 
                      maximumFractionDigits: 2 
                    })}
                  </td>
                  <td style={{ color: diffColor }}>
                    {isPositive ? '+' : ''}
                    {percentage.toFixed(2)}%
                  </td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
};

export default PredictionBenchmarkTable;
