import { CoinType, PriceAlert } from '../types';

// API response format for price alerts
interface ApiPriceAlert {
  alertCondition: 'ABOVE' | 'BELOW';
  id: string;
  isActive: boolean;
  monitoredPriceType: 'REAL' | 'PREDICTED';
  price: number;
}

// Map API response format to our app's PriceAlert type
const mapApiAlertToAppAlert = (apiAlert: ApiPriceAlert, coin: CoinType): PriceAlert => {
  return {
    id: apiAlert.id,
    coin: coin,
    type: apiAlert.monitoredPriceType === 'REAL' ? 'real-time' : 'predicted',
    condition: apiAlert.alertCondition === 'ABOVE' ? 'above' : 'below',
    threshold: apiAlert.price,
    active: apiAlert.isActive,
    createdAt: new Date().toISOString() // API doesn't provide creation date
  };
};

// Get alerts for a specific coin
export const getAlerts = async (coin: CoinType, userId: string = ''): Promise<PriceAlert[]> => {
  try {
    // Get userId from user object in localStorage if not provided or empty
    let effectiveUserId = userId;
    
    if (!effectiveUserId) {
      const userJson = localStorage.getItem('user');
      if (userJson) {
        try {
          const user = JSON.parse(userJson);
          effectiveUserId = user.id;
        } catch (e) {
          console.error('Error parsing user from localStorage:', e);
        }
      }
    }
    
    // Build the URL - only include userId param if we have a value
    let url = `http://localhost:3001/api/notification/${coin}`;
    if (effectiveUserId && effectiveUserId !== 'unknown') {
      url += `?userId=${effectiveUserId}`;
    }
    
    console.log('Fetching alerts with URL:', url); // Debug log
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch alerts: ${response.statusText}`);
    }
    
    const apiAlerts: ApiPriceAlert[] = await response.json();
    return apiAlerts.map(alert => mapApiAlertToAppAlert(alert, coin));
    
  } catch (error) {
    console.error('Error fetching alerts:', error);
    return [];
  }
};

// Toggle alert active status
export const toggleAlertStatus = async (alertId: string, isActive: boolean): Promise<boolean> => {
  try {
    const response = await fetch(`http://localhost:3001/api/notification/edit?notificationId=${alertId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ isActive })
    });
    
    return response.ok;
  } catch (error) {
    console.error('Error toggling alert status:', error);
    return false;
  }
};

// Create a new alert
export const createAlert = async (
  coin: CoinType,
  type: 'real-time' | 'predicted',
  condition: 'above' | 'below',
  threshold: number,
  userId: string = ''
): Promise<PriceAlert | null> => {
  try {
    // Get userId from user object in localStorage if not provided or empty
    let effectiveUserId = userId;
    
    if (!effectiveUserId) {
      const userJson = localStorage.getItem('user');
      if (userJson) {
        try {
          const user = JSON.parse(userJson);
          effectiveUserId = user.id;
        } catch (e) {
          console.error('Error parsing user from localStorage:', e);
        }
      }
    }
    
    // Build the URL with userId as query parameter
    let url = `http://localhost:3001/api/notification/add`;
    if (effectiveUserId && effectiveUserId !== 'unknown') {
      url += `?userId=${effectiveUserId}`;
    }
    
    // Format request body according to API requirements
    const alertData = {
      coin: coin,
      price: threshold,
      alertCondition: condition === 'above' ? 'ABOVE' : 'BELOW',
      monitoredPriceType: type === 'real-time' ? 'REAL' : 'PREDICTED',
      isActive: true
    };
    
    console.log('Creating alert with URL:', url);
    console.log('Alert data:', alertData);
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(alertData)
    });
    
    if (response.ok) {
      const data = await response.json();
      return mapApiAlertToAppAlert(data, coin);
    }
    
    return null;
  } catch (error) {
    console.error('Error creating alert:', error);
    return null;
  }
};

// Edit an existing alert
export const editAlert = async (
  alertId: string,
  coin: CoinType,
  type: 'real-time' | 'predicted',
  condition: 'above' | 'below',
  threshold: number
): Promise<PriceAlert | null> => {
  try {
    // Format request body according to API requirements
    const alertData = {
      coin: coin,
      price: threshold,
      alertCondition: condition === 'above' ? 'ABOVE' : 'BELOW',
      monitoredPriceType: type === 'real-time' ? 'REAL' : 'PREDICTED',
    };
    
    console.log('Editing alert with ID:', alertId);
    console.log('New alert data:', alertData);
    
    const response = await fetch(`http://localhost:3001/api/notification/edit?notificationId=${alertId}`, {
      method: 'PATCH',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(alertData)
    });
    
    if (response.ok) {
      const data = await response.json();
      return mapApiAlertToAppAlert(data, coin);
    }
    
    return null;
  } catch (error) {
    console.error('Error editing alert:', error);
    return null;
  }
};

// Delete an alert
export const deleteAlert = async (alertId: string): Promise<boolean> => {
  try {
    const response = await fetch(`http://localhost:3001/api/notification/?notificationId=${alertId}`, {
      method: 'DELETE'
    });
    
    return response.ok;
  } catch (error) {
    console.error('Error deleting alert:', error);
    return false;
  }
};

export default {
  getAlerts,
  toggleAlertStatus,
  createAlert,
  deleteAlert,
  editAlert
};
