import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { AuthState, LoginCredentials, RegisterCredentials, User } from '../types/auth';

interface AuthContextType extends AuthState {
  login: (credentials: LoginCredentials) => Promise<{ success: boolean; error?: string }>;
  register: (credentials: RegisterCredentials) => Promise<{ success: boolean; error?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: false
  });

  // Updated login function to handle the new response format
  const login = async (credentials: LoginCredentials) => {
    setAuthState(prev => ({ ...prev, isLoading: true }));
    
    try {
      const response = await fetch('http://localhost:3001/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: credentials.email,
          password: credentials.password
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
        return { 
          success: false, 
          error: data.error || data.message || 'Unknown error occurred' 
        };
      }
      
      // Parse user from the new response format
      const user: User = {
        id: data.user?.id || 'unknown',
        name: data.user?.name || '',
        email: credentials.email
      };
      
      setAuthState({
        user,
        isAuthenticated: true,
        isLoading: false
      });
      
      // Store authentication in localStorage
      localStorage.setItem('user', JSON.stringify(user));
      
      return { success: true };
    } catch (error) {
      console.error("Login error:", error);
      setAuthState(prev => ({ ...prev, isLoading: false }));
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'An error occurred while connecting to the server.' 
      };
    }
  };

  // Updated register function to include name and handle the new response format
  const register = async (credentials: RegisterCredentials) => {
    setAuthState(prev => ({ ...prev, isLoading: true }));
    
    try {
      if (credentials.password !== credentials.confirmPassword) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
        return { 
          success: false, 
          error: 'Passwords do not match.' 
        };
      }
      
      const response = await fetch('http://localhost:3001/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: credentials.name, // Include name in request
          email: credentials.email,
          password: credentials.password
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        setAuthState(prev => ({ ...prev, isLoading: false }));
        return { 
          success: false, 
          error: data.error || data.message || 'Registration failed' 
        };
      }
      
      // Parse user from the new response format
      const user: User = {
        id: data.user?.id || 'unknown',
        name: data.user?.name || credentials.name, // Use response name or fallback to provided name
        email: credentials.email
      };
      
      setAuthState({
        user,
        isAuthenticated: true,
        isLoading: false
      });
      
      // Store authentication in localStorage
      localStorage.setItem('user', JSON.stringify(user));
      
      return { success: true };
    } catch (error) {
      console.error("Registration error:", error);
      setAuthState(prev => ({ ...prev, isLoading: false }));
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'An error occurred while connecting to the server.' 
      };
    }
  };

  const logout = () => {
    setAuthState({
      user: null,
      isAuthenticated: false,
      isLoading: false
    });
    localStorage.removeItem('user');
  };

  // Check for existing session on component mount
  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      try {
        const user = JSON.parse(storedUser);
        setAuthState({
          user,
          isAuthenticated: true,
          isLoading: false
        });
      } catch (error) {
        console.error("Error parsing stored user:", error);
        localStorage.removeItem('user');
      }
    }
  }, []);

  return (
    <AuthContext.Provider value={{ ...authState, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
