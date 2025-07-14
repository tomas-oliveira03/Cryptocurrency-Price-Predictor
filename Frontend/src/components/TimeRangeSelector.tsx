import { TimeRange } from '../types';
import { motion } from 'framer-motion';

interface TimeRangeSelectorProps {
  selectedTimeRange: TimeRange;
  onChange: (timeRange: TimeRange) => void;
}

const TimeRangeSelector: React.FC<TimeRangeSelectorProps> = ({ selectedTimeRange, onChange }) => {
  const timeRanges: { value: TimeRange; label: string }[] = [
    { value: '7days', label: '7D' },
    { value: '30days', label: '1M' },
    { value: '1year', label: '1Y' },
  ];

  return (
    <>
      {timeRanges.map((range) => (
        <motion.button
          key={range.value}
          className={selectedTimeRange === range.value ? 'active' : ''}
          onClick={() => onChange(range.value)}
          whileHover={{ 
            scale: selectedTimeRange === range.value ? 1.02 : 1.05,
            color: selectedTimeRange === range.value ? '#4facfe' : '#ffffff'
          }}
          whileTap={{ scale: 0.95 }}
          style={{
            color: selectedTimeRange === range.value ? '#4facfe' : '#a0a0a0',
            backgroundColor: selectedTimeRange === range.value ? 'rgba(79, 172, 254, 0.15)' : 'transparent',
            borderColor: selectedTimeRange === range.value ? 'rgba(79, 172, 254, 0.3)' : 'transparent'
          }}
        >
          {range.label}
        </motion.button>
      ))}
    </>
  );
};

export default TimeRangeSelector;
