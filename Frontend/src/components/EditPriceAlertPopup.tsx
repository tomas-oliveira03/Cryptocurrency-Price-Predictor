import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CoinType, PriceAlert } from '../types';
import notificationService from '../services/notificationService';

interface EditPriceAlertPopupProps {
  isOpen: boolean;
  onClose: () => void;
  alert: PriceAlert | null;
  currentPrice: number;
}

const EditPriceAlertPopup: React.FC<EditPriceAlertPopupProps> = ({
  isOpen,
  onClose,
  alert,
  currentPrice
}) => {
  // State for form fields
  const [notificationType, setNotificationType] = useState<'real-time' | 'predicted'>('real-time');
  const [condition, setCondition] = useState<'above' | 'below'>('above');
  const [threshold, setThreshold] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Keep track of initial values to compare changes
  const [initialValues, setInitialValues] = useState({
    type: 'real-time',
    condition: 'above',
    threshold: ''
  });

  // Track if form has been modified
  const hasChanges = () => {
    return notificationType !== initialValues.type ||
           condition !== initialValues.condition ||
           parseFloat(threshold) !== parseFloat(initialValues.threshold);
  };
  
  // Initialize form with alert data when popup opens
  useEffect(() => {
    if (isOpen && alert) {
      setNotificationType(alert.type);
      setCondition(alert.condition);
      setThreshold(alert.threshold.toString());
      
      // Store initial values for comparison
      setInitialValues({
        type: alert.type,
        condition: alert.condition,
        threshold: alert.threshold.toString()
      });
    }
  }, [isOpen, alert]);
  
  // Handle clicking outside to close popup
  const popupRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (popupRef.current && !popupRef.current.contains(event.target as Node)) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen, onClose]);

  // Handle form submission
  const handleSubmit = async () => {
    if (!alert) return;
    
    // Validate input
    if (!threshold || isNaN(parseFloat(threshold))) {
      setError('Please enter a valid price threshold');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    
    try {
      const result = await notificationService.editAlert(
        alert.id,
        alert.coin,
        notificationType,
        condition,
        parseFloat(threshold)
      );
      
      if (result) {
        onClose();
      } else {
        throw new Error('Failed to update alert');
      }
    } catch (err) {
      console.error('Error updating alert:', err);
      setError('An unexpected error occurred. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleNotificationTypeChange = (type: 'real-time' | 'predicted') => {
    setNotificationType(type);
    if (error) setError(null);
  };

  const handleConditionChange = (cond: 'above' | 'below') => {
    setCondition(cond);
    if (error) setError(null);
  };

  const handleThresholdChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Limit to 2 decimal places
    const value = e.target.value;
    
    // Check if the value has more than 2 decimal places
    const decimalParts = value.split('.');
    if (decimalParts.length > 1 && decimalParts[1].length > 2) {
      // Truncate to 2 decimal places
      setThreshold(Number(value).toFixed(2));
    } else {
      setThreshold(value);
    }
    
    // Clear error when input changes
    if (error) setError(null);
  };

  return (
    <AnimatePresence>
      {isOpen && alert && (
        <motion.div 
          className="price-alert-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div 
            className="price-alert-popup"
            ref={popupRef}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="price-alert-header">
              <h3>Edit Alert for {alert.coin}</h3>
              <button 
                className="close-button"
                onClick={onClose}
              >
                ×
              </button>
            </div>
            
            <div className="price-alert-content">
              <div className="alert-section">
                <label>Notification Type</label>
                <div className="toggle-buttons">
                  <button 
                    className={notificationType === 'real-time' ? 'active' : ''}
                    onClick={() => handleNotificationTypeChange('real-time')}
                  >
                    ⚡ Real-time Price
                  </button>
                  <button 
                    className={notificationType === 'predicted' ? 'active' : ''}
                    onClick={() => handleNotificationTypeChange('predicted')}
                  >
                    🔮 Predicted Price
                  </button>
                </div>
              </div>
              
              <div className="alert-section">
                <label>Notify me when price is:</label>
                <div className="toggle-buttons">
                  <button 
                    className={condition === 'below' ? 'active' : ''}
                    onClick={() => handleConditionChange('below')}
                  >
                    📉 Below
                  </button>
                  <button 
                    className={condition === 'above' ? 'active' : ''}
                    onClick={() => handleConditionChange('above')}
                  >
                    📈 Above
                  </button>
                </div>
              </div>
              
              <div className="alert-section">
                <label>Price Threshold ($)</label>
                <input 
                  type="number" 
                  value={threshold}
                  onChange={handleThresholdChange}
                  step="0.01"
                  placeholder="Enter price threshold"
                />
              </div>
              
              <div className="current-price-info">
                Current price: <strong>${currentPrice.toLocaleString()}</strong>
              </div>

              {error && (
                <motion.div 
                  className="alert-error-message"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  {error}
                </motion.div>
              )}
            </div>
            
            <div className="price-alert-actions">
              <button 
                className="cancel-button"
                onClick={onClose}
                disabled={isSubmitting}
              >
                Cancel
              </button>
              <button 
                className="set-alert-button"
                onClick={handleSubmit}
                disabled={isSubmitting || !hasChanges()}
              >
                {isSubmitting ? 'Updating...' : 'Update Alert'}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default EditPriceAlertPopup;
