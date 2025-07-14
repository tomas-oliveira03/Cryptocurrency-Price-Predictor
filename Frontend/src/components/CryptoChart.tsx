import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { CryptoData, DataField } from '../types';
import { motion } from 'framer-motion';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface CryptoChartProps {
  data: CryptoData;
  selectedFields: DataField[];
}

const CryptoChart: React.FC<CryptoChartProps> = ({ data, selectedFields }) => {
  // Parse dates safely and format them consistently for display
  const formatDate = (dateStr: string): string => {
    try {
      const date = new Date(dateStr);
      if (isNaN(date.getTime())) {
        throw new Error('Invalid date');
      }
      return date.toLocaleDateString();
    } catch (e) {
      console.warn("Invalid date format in chart:", dateStr);
      return dateStr; // Return the original string if parsing fails
    }
  };

  // Prepare all dates from all selected datasets
  const allDates = new Set<string>();
  
  if (selectedFields.includes('historical_price')) {
    data.historical_price.forEach(item => allDates.add(item.date));
  }
  
  if (selectedFields.includes('predicted_price')) {
    data.predicted_price.forEach(item => allDates.add(item.date));
  }
  
  if (selectedFields.includes('positive_sentiment_ratio')) {
    data.positive_sentiment_ratio.forEach(item => allDates.add(item.date));
  }
  
  // Sort dates chronologically
  const sortedDates = Array.from(allDates).sort((a, b) => {
    try {
      return new Date(a).getTime() - new Date(b).getTime();
    } catch (e) {
      return 0;
    }
  });
  
  // These are the colors we'll use consistently for each data type
  const colorPalette = {
    historicalPrice: 'rgb(53, 162, 235)',
    predictedPrice: 'rgb(255, 140, 0)', // Changed from red to orange
    sentiment: 'rgba(179, 136, 255, 0.8)', // Light purple
    neutral: 'rgba(255, 255, 255, 0.5)'
  };
  
  // Prepare datasets with enhanced styling
  const datasets = [];
  
  if (selectedFields.includes('historical_price')) {
    datasets.push({
      label: 'Historical Price',
      data: sortedDates.map(date => {
        const point = data.historical_price.find(item => item.date === date);
        return point ? point.price : null;
      }),
      borderColor: colorPalette.historicalPrice,
      backgroundColor: 'rgba(53, 162, 235, 0.5)',
      borderWidth: 2,
      yAxisID: 'y',
      pointRadius: sortedDates.length > 60 ? 0 : 3,
      pointHoverRadius: 5,
      tension: 0.4,
    });

    // Add a reference line at the initial price
    if (data.historical_price.length > 0) {
      // Find the earliest date in the sorted dates that has historical price data
      const initialDateIndex = sortedDates.findIndex(date => 
        data.historical_price.some(item => item.date === date)
      );
      
      if (initialDateIndex >= 0) {
        const initialDate = sortedDates[initialDateIndex];
        const initialPricePoint = data.historical_price.find(item => item.date === initialDate);
        const initialPrice = initialPricePoint ? initialPricePoint.price : null;
        
        if (initialPrice !== null) {
          datasets.push({
            label: 'Initial Price',
            data: sortedDates.map(() => initialPrice),
            borderColor: 'rgba(255, 215, 0, 0.7)', // Yellow color
            backgroundColor: 'transparent',
            borderWidth: 1,
            borderDash: [3, 3],
            yAxisID: 'y',
            pointRadius: 0,
            pointHoverRadius: 0,
            tension: 0,
            fill: false,
          });
        }
      }
    }
  }
  
  if (selectedFields.includes('predicted_price')) {
    // Find the last historical price point to ensure connection
    let lastHistoricalDate = null;
    let lastHistoricalPrice = null;
    
    if (data.historical_price.length > 0) {
      // Sort historical prices by date to find the last one
      const sortedHistoricalPrices = [...data.historical_price].sort((a, b) => {
        return new Date(b.date).getTime() - new Date(a.date).getTime();
      });
      
      lastHistoricalDate = sortedHistoricalPrices[0].date;
      lastHistoricalPrice = sortedHistoricalPrices[0].price;
    }
    
    datasets.push({
      label: 'Predicted Price',
      data: sortedDates.map(date => {
        // If this is the last historical date, use that price for connection
        if (date === lastHistoricalDate && lastHistoricalPrice !== null) {
          return lastHistoricalPrice;
        }
        
        const point = data.predicted_price.find(item => item.date === date);
        return point ? point.price : null;
      }),
      borderColor: colorPalette.predictedPrice,
      backgroundColor: 'rgba(255, 140, 0, 0.5)', // Changed to match the orange color
      borderWidth: 2,
      yAxisID: 'y',
      pointRadius: sortedDates.length > 60 ? 0 : 3,
      pointHoverRadius: 5,
      tension: 0.3,
    });
  }
  
  if (selectedFields.includes('positive_sentiment_ratio')) {
    datasets.push({
      label: 'Sentiment Ratio',
      data: sortedDates.map(date => {
        // Check if this date has historical price data
        const hasHistoricalData = data.historical_price.some(item => item.date === date);
        // Only return sentiment for dates with historical data
        if (hasHistoricalData) {
          const point = data.positive_sentiment_ratio.find(item => item.date === date);
          return point && point.sentiment > 0 && point.sentiment < 1 ? point.sentiment * 100 : 50;
        }
        // Return null for predicted dates (will not be displayed)
        return null;
      }),
      borderColor: colorPalette.sentiment,
      backgroundColor: 'rgba(179, 136, 255, 0.2)',
      pointBackgroundColor: colorPalette.sentiment, // Make sure points use the same color
      pointBorderColor: colorPalette.sentiment,     // Make sure point borders use the same color
      fill: true,
      borderWidth: 2,
      yAxisID: 'y1', 
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.4,
    });
    
    // Add a reference line at 50% sentiment (neutral)
    datasets.push({
      label: 'Neutral Sentiment',
      data: sortedDates.map(() => 50), // Show for all dates to make it reach the end
      borderColor: colorPalette.neutral,
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderDash: [3, 3],
      yAxisID: 'y1',
      pointRadius: 0,
      pointHoverRadius: 0,
      tension: 0,
      fill: false,
      hidden: false
    });
  }
  
  const chartData = {
    labels: sortedDates.map(formatDate),
    datasets,
  };
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    stacked: false,
    plugins: {
      title: {
        display: true,
        text: `Price and Sentiment Analysis`,
        font: {
          size: 18,
          family: "'Inter', sans-serif",
          weight: '600'
        },
        padding: {
          bottom: 20
        },
        color: '#e0e0e0'
      },
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true, // Use the pointStyle from the dataset
          padding: 20,
          font: {
            family: "'Inter', sans-serif",
            size: 12
          },
          color: '#e0e0e0'
        },
        // Enable legend interaction for toggling datasets
        onClick: function(_: any, legendItem: any, legend: any) {
          const index = legendItem.datasetIndex;
          const ci = legend.chart;
          
          if (ci.isDatasetVisible(index)) {
            ci.hide(index);
            legendItem.hidden = true;
          } else {
            ci.show(index);
            legendItem.hidden = false;
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(15, 15, 26, 0.8)',
        titleFont: {
          family: "'Inter', sans-serif",
          size: 14
        },
        bodyFont: {
          family: "'Inter', sans-serif",
          size: 13
        },
        padding: 12,
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        // Filter out reference lines from tooltips
        filter: function(tooltipItem: any) {
          // Skip showing tooltip for reference lines
          const label = tooltipItem.dataset.label;
          return label !== 'Initial Price' && label !== 'Neutral Sentiment';
        },
        callbacks: {
          label: function(context: any) {
            let label = context.dataset.label || '';
            
            // Skip showing Predicted Price at the connection point (last historical date)
            if (label === 'Predicted Price') {
              // Get the data point's date
              const pointIndex = context.dataIndex;
              const date = sortedDates[pointIndex];
              
              // Check if this is the last historical date
              const isLastHistoricalDate = data.historical_price.some(item => 
                item.date === date && 
                !data.historical_price.some(h => new Date(h.date) > new Date(date))
              );
              
              if (isLastHistoricalDate) {
                return null; // Don't show predicted price at the connection point
              }
            }
            
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              if (context.dataset.yAxisID === 'y1') {
                // Sentiment values - no span
                label += `${context.parsed.y.toFixed(2)}%`;
              } else {
                // Price values - no span
                label += `$${context.parsed.y.toLocaleString()}`;
              }
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: {
          maxTicksLimit: sortedDates.length > 60 ? 10 : 20,
          font: {
            family: "'Inter', sans-serif",
            size: 11
          },
          color: '#a0a0a0'
        },
        grid: {
          display: true,
          color: 'rgba(255, 255, 255, 0.05)'
        }
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: {
          display: true,
          text: 'Price (USD)',
          font: {
            family: "'Inter', sans-serif",
            size: 12,
          },
          color: colorPalette.historicalPrice
        },
        ticks: {
          font: {
            family: "'Inter', sans-serif",
            size: 11
          },
          color: colorPalette.historicalPrice,
          callback: (value: number) => '$' + value.toLocaleString()
        },
        grid: {
          color: 'rgba(255, 255, 255, 0.05)'
        }
      },
      y1: {
        type: 'linear' as const,
        display: selectedFields.includes('positive_sentiment_ratio'),
        position: 'right' as const,
        min: 0,
        max: 100,
        title: {
          display: true,
          text: 'Sentiment Ratio (%)',
          font: {
            family: "'Inter', sans-serif",
            size: 12,
          },
          color: colorPalette.sentiment
        },
        ticks: {
          font: {
            family: "'Inter', sans-serif",
            size: 11
          },
          color: colorPalette.sentiment,
          callback: (value: number) => value.toFixed(0) + '%'
        },
        grid: {
          display: false
        }
      },
    },
  };

  return (
    <motion.div 
      className="chart-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <Line data={chartData} options={options} />
    </motion.div>
  );
};

export default CryptoChart;
