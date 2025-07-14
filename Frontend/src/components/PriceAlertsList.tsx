import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { PriceAlert, CoinType } from '../types';
import notificationService from '../services/notificationService';
import EditPriceAlertPopup from './EditPriceAlertPopup';

interface PriceAlertsListProps {
  isOpen: boolean;
  onClose: () => void;
  onAddNew: () => void;
  currentCoin: CoinType;
  currentPrice: number;
  refreshTrigger?: number;
}

const PriceAlertsList: React.FC<PriceAlertsListProps> = ({
  isOpen,
  onClose,
  onAddNew,
  currentCoin,
  currentPrice,
  refreshTrigger = 0
}) => {
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingAlert, setEditingAlert] = useState<PriceAlert | null>(null);
  
  // Add filter states
  const [filters, setFilters] = useState({
    status: 'all', // 'all', 'active', 'inactive'
    type: 'all',   // 'all', 'real-time', 'predicted'
    condition: 'all' // 'all', 'above', 'below'
  });
  
  // Fetch alerts when component mounts, coin changes, or refreshTrigger changes
  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      notificationService.getAlerts(currentCoin)
        .then(fetchedAlerts => {
          // Sort alerts by threshold value, and when equal put 'below' condition first
          const sortedAlerts = [...fetchedAlerts].sort((a, b) => {
            // First compare thresholds
            if (a.threshold !== b.threshold) {
              return a.threshold - b.threshold;
            }
            // If thresholds are equal, put 'below' before 'above'
            return a.condition === 'below' ? -1 : 1;
          });
          setAlerts(sortedAlerts);
          setLoading(false);
        });
    }
  }, [isOpen, currentCoin, refreshTrigger]);
  
  // Apply filters to alerts
  const filteredAlerts = alerts.filter(alert => {
    // Filter by status
    if (filters.status !== 'all') {
      if (filters.status === 'active' && !alert.active) return false;
      if (filters.status === 'inactive' && alert.active) return false;
    }
    
    // Filter by type
    if (filters.type !== 'all' && alert.type !== filters.type) return false;
    
    // Filter by condition
    if (filters.condition !== 'all' && alert.condition !== filters.condition) return false;
    
    return true;
  });

  // Handle filter changes
  const handleFilterChange = (filterType: 'status' | 'type' | 'condition', value: string) => {
    setFilters(prevFilters => ({
      ...prevFilters,
      [filterType]: value
    }));
  };
  
  // Toggle alert status
  const handleToggleStatus = (alertId: string, currentStatus: boolean) => {
    // 1. Immediately update UI state (optimistic update)
    const newStatus = !currentStatus;
    setAlerts(alerts.map(alert => 
      alert.id === alertId ? { ...alert, active: newStatus } : alert
    ));
    
    // 2. Make API call in background
    notificationService.toggleAlertStatus(alertId, newStatus)
      .then(success => {
        // 3. If the API call fails, revert the UI change
        if (!success) {
          console.error('Failed to update alert status on server');
          setAlerts(alerts.map(alert => 
            alert.id === alertId ? { ...alert, active: currentStatus } : alert
          ));
        }
      });
  };
  
  // Delete alert
  const handleDeleteAlert = async (alertId: string) => {
      const success = await notificationService.deleteAlert(alertId);
      
      if (success) {
        setAlerts(alerts.filter(alert => alert.id !== alertId));
      }
  };

  // Handle edit button click
  const handleEditAlert = (alert: PriceAlert) => {
    setEditingAlert(alert);
  };
  
  // Handle close edit popup
  const handleCloseEditPopup = () => {
    setEditingAlert(null);
    // Re-fetch alerts to get updated data and sort them
    notificationService.getAlerts(currentCoin)
      .then(fetchedAlerts => {
        const sortedAlerts = [...fetchedAlerts].sort((a, b) => a.threshold - b.threshold);
        setAlerts(sortedAlerts);
      });
  };
  
  return (
    <>
      <AnimatePresence>
        {isOpen && (
          <motion.div 
            className="price-alert-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div 
              className="price-alert-popup"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="price-alert-header">
                <h3>Price Alerts for {currentCoin}</h3>
                <button className="close-button" onClick={onClose}>×</button>
              </div>
              
              <div className="price-alert-content alerts-list-content">

                {/* Filters section */}
                {!loading && alerts.length > 0 && (
                  <div className="alerts-filter-section">
                    <div className="filter-row">
                      <div className="filter-group">
                        <label className="filter-label">Status:</label>
                        <div className="filter-options">
                          <button 
                            className={filters.status === 'all' ? 'active' : ''}
                            onClick={() => handleFilterChange('status', 'all')}
                          >
                            All
                          </button>
                          <button 
                            className={filters.status === 'active' ? 'active' : ''}
                            onClick={() => handleFilterChange('status', 'active')}
                          >
                            Active
                          </button>
                          <button 
                            className={filters.status === 'inactive' ? 'active' : ''}
                            onClick={() => handleFilterChange('status', 'inactive')}
                          >
                            Inactive
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className="filter-row">
                      <div className="filter-group">
                        <label className="filter-label">Type:</label>
                        <div className="filter-options">
                          <button 
                            className={filters.type === 'all' ? 'active' : ''}
                            onClick={() => handleFilterChange('type', 'all')}
                          >
                            All
                          </button>
                          <button 
                            className={filters.type === 'real-time' ? 'active' : ''}
                            onClick={() => handleFilterChange('type', 'real-time')}
                          >
                            ⚡ Real-time
                          </button>
                          <button 
                            className={filters.type === 'predicted' ? 'active' : ''}
                            onClick={() => handleFilterChange('type', 'predicted')}
                          >
                            🔮 Predicted
                          </button>
                        </div>
                      </div>
                    </div>

                    <div className="filter-row">
                      <div className="filter-group">
                        <label className="filter-label">Condition:</label>
                        <div className="filter-options">
                          <button 
                            className={filters.condition === 'all' ? 'active' : ''}
                            onClick={() => handleFilterChange('condition', 'all')}
                          >
                            All
                          </button>
                          <button 
                            className={filters.condition === 'below' ? 'active' : ''}
                            onClick={() => handleFilterChange('condition', 'below')}
                          >
                            📉 Below
                          </button>
                          <button 
                            className={filters.condition === 'above' ? 'active' : ''}
                            onClick={() => handleFilterChange('condition', 'above')}
                          >
                            📈 Above
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <div className="current-price-info">
                  Current price: <strong>${currentPrice.toLocaleString()}</strong>
                </div>
                
                {loading ? (
                  <div className="loading-message">Loading your alerts...</div>
                ) : (
                  <>
                    {alerts.length === 0 ? (
                      <div className="no-alerts-message">
                        <p>You don't have any alerts set for {currentCoin}.</p>
                      </div>
                    ) : (
                      <div className="alerts-list">
                        {filteredAlerts.length === 0 ? (
                          <div className="no-alerts-message">
                            <p>No alerts match your current filters.</p>
                          </div>
                        ) : (
                          filteredAlerts.map(alert => (
                            <div 
                              key={alert.id} 
                              className={`alert-item ${!alert.active ? 'inactive' : ''}`}
                            >
                              <button 
                                className="edit-button"
                                onClick={() => handleEditAlert(alert)}
                                title="Edit alert"
                              >
                                ✏️
                              </button>
                              <div className="alert-info">
                                <div className="alert-type">
                                  {alert.type === 'real-time' ? '⚡ Real-time' : '🔮 Predicted'}
                                </div>
                                <div className="alert-condition">
                                  {alert.condition === 'above' ? '📈' : '📉'} Price {alert.condition} <strong>${alert.threshold.toLocaleString()}</strong>
                                </div>
                              </div>
                              <div className="alert-actions">
                                <label className="toggle">
                                  <input 
                                    type="checkbox" 
                                    checked={alert.active}
                                    onChange={() => handleToggleStatus(alert.id, alert.active)}
                                  />
                                  <span className="slider"></span>
                                </label>
                                <button 
                                  className="delete-button"
                                  onClick={() => handleDeleteAlert(alert.id)}
                                  title="Delete alert"
                                >
                                  🗑️
                                </button>
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
              
              <div className="price-alert-actions">
                <button className="cancel-button" onClick={onClose}>Close</button>
                <button className="set-alert-button" onClick={onAddNew}>+ New Alert</button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <EditPriceAlertPopup 
        isOpen={editingAlert !== null}
        onClose={handleCloseEditPopup}
        alert={editingAlert}
        currentPrice={currentPrice}
      />
    </>
  );
};

export default PriceAlertsList;
