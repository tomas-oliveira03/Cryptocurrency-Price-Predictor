import { DataField } from '../types';
import { motion } from 'framer-motion';

interface FieldSelectorProps {
  selectedFields: DataField[];
  onChange: (fields: DataField[]) => void;
}

const FieldSelector: React.FC<FieldSelectorProps> = ({ selectedFields, onChange }) => {
  const fields: { value: DataField; label: string; icon: string }[] = [
    { value: 'historical_price', label: 'Historical Price', icon: '📊' },
    { value: 'predicted_price', label: 'Predicted Price', icon: '🔮' },
    { value: 'positive_sentiment_ratio', label: 'Positive Sentiment', icon: '😀' },
  ];

  const handleCheckboxChange = (field: DataField) => {
    if (selectedFields.includes(field)) {
      onChange(selectedFields.filter(f => f !== field));
    } else {
      onChange([...selectedFields, field]);
    }
  };

  return (
    <motion.div 
      className="field-selector"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      <span>Select Data to Display</span>
      <div className="checkbox-group">
        {fields.map((field, index) => (
          <motion.div 
            key={field.value} 
            className="checkbox-item"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.3 + index * 0.1 }}
            whileHover={{ scale: 1.05, x: 5 }}
          >
            <input
              type="checkbox"
              id={`field-${field.value}`}
              checked={selectedFields.includes(field.value)}
              onChange={() => handleCheckboxChange(field.value)}
            />
            <label htmlFor={`field-${field.value}`}>
              <span style={{ marginRight: '6px' }}>{field.icon}</span>
              {field.label}
            </label>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

export default FieldSelector;
