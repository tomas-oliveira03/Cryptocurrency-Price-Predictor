import { ModelBenchmarks as ModelBenchmarksType } from '../types';
import { motion } from 'framer-motion';

interface ModelBenchmarksProps {
  benchmarks: ModelBenchmarksType;
}

// Helper function to format potentially large numbers
const formatNumber = (value: number, decimalPlaces: number): string => {
  // Use scientific notation for very large numbers
  if (value > 1000000) {
    return value.toExponential(2);
  }
  return value.toFixed(decimalPlaces);
};

const ModelBenchmarks: React.FC<ModelBenchmarksProps> = ({ benchmarks }) => {
  return (
    <motion.div 
      className="model-benchmarks"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3>Model Performance Metrics</h3>
      <div className="benchmarks-grid">
        {[
          {
            label: "MAE",
            value: formatNumber(benchmarks.mae, 2),
            desc: "Mean Absolute Error",
            color: "#4facfe"
          },
          {
            label: "MAPE",
            value: `${formatNumber(benchmarks.mape, 2)}%`,
            desc: "Mean Absolute Percentage Error",
            color: "#00f2fe"
          },
          {
            label: "MSE",
            value: formatNumber(benchmarks.mse, 2),
            desc: "Mean Square Error",
            color: "#0ae2f1"
          },
          {
            label: "RMSE",
            value: formatNumber(benchmarks.rmse, 2),
            desc: "Root Mean Square Error",
            color: "#19c3e0"
          },
          {
            label: "R²",
            value: formatNumber(benchmarks.r2, 4),
            desc: "Coefficient of Determination",
            color: "#28a5cf"
          }
        ].map((metric, index) => (
          <motion.div 
            className="benchmark-item"
            key={metric.label}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
            whileHover={{ 
              scale: 1.05,
              boxShadow: "0 15px 30px rgba(0, 0, 0, 0.3)",
              borderColor: `${metric.color}50`
            }}
          >
            <div className="benchmark-label" style={{ color: metric.color }}>{metric.label}</div>
            <div className="benchmark-value" title={metric.value}>{metric.value}</div>
            <div className="benchmark-desc">{metric.desc}</div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

export default ModelBenchmarks;
