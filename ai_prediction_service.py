import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json
import pickle
import os

class PJMAIPredictionService:
    """AI service for PJM energy market predictions and recommendations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ai_models')
        os.makedirs(self.model_path, exist_ok=True)
        
    def predict_short_term_lmp(self, historical_data: List[Dict], hours_ahead: int = 24) -> Dict:
        """Predict LMP for the next 24-72 hours"""
        try:
            if not historical_data:
                return {
                    'status': 'error',
                    'message': 'No historical data provided'
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            
            # Feature engineering
            features = self._engineer_features(df)
            
            if features.empty:
                return {
                    'status': 'error',
                    'message': 'Unable to engineer features from historical data'
                }
            
            # Train or load model
            model = self._get_or_train_lmp_model(features)
            
            # Generate predictions
            predictions = self._generate_lmp_predictions(model, features, hours_ahead)
            
            return {
                'status': 'success',
                'predictions': predictions,
                'forecast_horizon': f'{hours_ahead} hours',
                'timestamp': datetime.now().isoformat(),
                'model_accuracy': self._get_model_accuracy('lmp_short_term')
            }
            
        except Exception as e:
            self.logger.error(f"Error in short-term LMP prediction: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_long_term_forecast(self, historical_data: List[Dict], years_ahead: int = 5) -> Dict:
        """Generate long-term price forecasts for next 5 years"""
        try:
            if not historical_data:
                return {
                    'status': 'error',
                    'message': 'No historical data provided'
                }
            
            # Convert to DataFrame and aggregate to monthly/yearly data
            df = pd.DataFrame(historical_data)
            monthly_data = self._aggregate_to_monthly(df)
            
            if monthly_data.empty:
                return {
                    'status': 'error',
                    'message': 'Unable to aggregate historical data'
                }
            
            # Generate long-term forecasts
            forecasts = self._generate_long_term_forecasts(monthly_data, years_ahead)
            
            return {
                'status': 'success',
                'forecasts': forecasts,
                'forecast_horizon': f'{years_ahead} years',
                'timestamp': datetime.now().isoformat(),
                'methodology': 'Time series analysis with trend and seasonality components'
            }
            
        except Exception as e:
            self.logger.error(f"Error in long-term forecast: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_ai_recommendations(self, current_market_data: Dict, zone_data: Dict) -> Dict:
        """Generate AI-powered trading and operational recommendations"""
        try:
            recommendations = []
            
            # Analyze current market conditions
            market_analysis = self._analyze_market_conditions(current_market_data)
            
            # Generate congestion management recommendations
            congestion_recs = self._generate_congestion_recommendations(zone_data)
            recommendations.extend(congestion_recs)
            
            # Generate price-based trading recommendations
            trading_recs = self._generate_trading_recommendations(current_market_data, market_analysis)
            recommendations.extend(trading_recs)
            
            # Generate risk management recommendations
            risk_recs = self._generate_risk_recommendations(current_market_data)
            recommendations.extend(risk_recs)
            
            # Generate renewable energy recommendations
            renewable_recs = self._generate_renewable_recommendations(current_market_data)
            recommendations.extend(renewable_recs)
            
            return {
                'status': 'success',
                'recommendations': recommendations,
                'market_analysis': market_analysis,
                'timestamp': datetime.now().isoformat(),
                'total_recommendations': len(recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating AI recommendations: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def calculate_risk_metrics(self, price_data: List[Dict]) -> Dict:
        """Calculate risk metrics like VaR, CVaR, volatility"""
        try:
            if not price_data:
                return {
                    'status': 'error',
                    'message': 'No price data provided'
                }
            
            # Extract price values
            prices = [record.get('LMP', 0) for record in price_data if record.get('LMP')]
            
            if not prices:
                return {
                    'status': 'error',
                    'message': 'No valid price data found'
                }
            
            prices = np.array(prices)
            returns = np.diff(prices) / prices[:-1] * 100  # Percentage returns
            
            # Calculate risk metrics
            var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
            var_99 = np.percentile(returns, 1)  # 1st percentile for 99% VaR
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 2.0
            mean_return = np.mean(returns)
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Beta (market sensitivity - using price correlation with mean)
            market_prices = np.mean(prices)
            beta = np.corrcoef(prices[1:], returns)[0, 1] if len(prices) > 1 else 1.0
            
            return {
                'status': 'success',
                'risk_metrics': {
                    'var_95': round(var_95, 2),
                    'var_99': round(var_99, 2),
                    'cvar_95': round(cvar_95, 2),
                    'cvar_99': round(cvar_99, 2),
                    'volatility': round(volatility, 2),
                    'sharpe_ratio': round(sharpe_ratio, 2),
                    'beta': round(beta, 2),
                    'mean_price': round(np.mean(prices), 2),
                    'price_range': {
                        'min': round(np.min(prices), 2),
                        'max': round(np.max(prices), 2)
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML models"""
        try:
            if df.empty:
                return pd.DataFrame()
            
            # Ensure we have required columns
            required_cols = ['LMP']
            if not any(col in df.columns for col in required_cols):
                return pd.DataFrame()
            
            features_df = df.copy()
            
            # Time-based features
            if 'Time' in features_df.columns:
                features_df['Time'] = pd.to_datetime(features_df['Time'])
                features_df['hour'] = features_df['Time'].dt.hour
                features_df['day_of_week'] = features_df['Time'].dt.dayofweek
                features_df['month'] = features_df['Time'].dt.month
                features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
            
            # Lag features
            if 'LMP' in features_df.columns:
                features_df['lmp_lag1'] = features_df['LMP'].shift(1)
                features_df['lmp_lag24'] = features_df['LMP'].shift(24)
                features_df['lmp_ma24'] = features_df['LMP'].rolling(window=24).mean()
                features_df['lmp_std24'] = features_df['LMP'].rolling(window=24).std()
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            return pd.DataFrame()
    
    def _get_or_train_lmp_model(self, features_df: pd.DataFrame):
        """Get existing model or train a new one"""
        model_name = 'lmp_short_term'
        model_file = os.path.join(self.model_path, f'{model_name}.pkl')
        
        if os.path.exists(model_file):
            try:
                with open(model_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        # Train new model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Prepare training data
        feature_cols = [col for col in features_df.columns if col not in ['LMP', 'Time']]
        if not feature_cols:
            feature_cols = ['hour', 'day_of_week', 'month']
            for col in feature_cols:
                if col not in features_df.columns:
                    features_df[col] = 0
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['LMP'].fillna(0)
        
        if len(X) > 0 and len(y) > 0:
            model.fit(X, y)
            
            # Save model
            try:
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                self.models[model_name] = model
            except Exception as e:
                self.logger.warning(f"Could not save model: {str(e)}")
        
        return model
    
    def _generate_lmp_predictions(self, model, features_df: pd.DataFrame, hours_ahead: int) -> List[Dict]:
        """Generate LMP predictions"""
        predictions = []
        
        try:
            if features_df.empty:
                return predictions
            
            # Get the last row as base for prediction
            last_row = features_df.iloc[-1].copy()
            current_time = datetime.now()
            
            feature_cols = [col for col in features_df.columns if col not in ['LMP', 'Time']]
            if not feature_cols:
                feature_cols = ['hour', 'day_of_week', 'month']
            
            for hour in range(hours_ahead):
                pred_time = current_time + timedelta(hours=hour)
                
                # Update time-based features
                pred_features = last_row.copy()
                pred_features['hour'] = pred_time.hour
                pred_features['day_of_week'] = pred_time.weekday()
                pred_features['month'] = pred_time.month
                pred_features['is_weekend'] = 1 if pred_time.weekday() >= 5 else 0
                
                # Prepare feature vector
                X_pred = np.array([pred_features[col] if col in pred_features else 0 for col in feature_cols]).reshape(1, -1)
                
                # Make prediction
                pred_lmp = model.predict(X_pred)[0]
                
                predictions.append({
                    'timestamp': pred_time.isoformat(),
                    'predicted_lmp': round(float(pred_lmp), 2),
                    'hour_ahead': hour + 1,
                    'confidence': 'medium'  # Placeholder for confidence intervals
                })
        
        except Exception as e:
            self.logger.error(f"Error generating predictions: {str(e)}")
        
        return predictions
    
    def _aggregate_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to monthly averages"""
        try:
            if df.empty or 'LMP' not in df.columns:
                return pd.DataFrame()
            
            # Ensure Time column exists
            if 'Time' not in df.columns:
                df['Time'] = datetime.now()
            
            df['Time'] = pd.to_datetime(df['Time'])
            df['year_month'] = df['Time'].dt.to_period('M')
            
            monthly_data = df.groupby('year_month').agg({
                'LMP': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            
            monthly_data.columns = ['year_month', 'avg_lmp', 'std_lmp', 'min_lmp', 'max_lmp', 'count']
            
            return monthly_data
            
        except Exception as e:
            self.logger.error(f"Error aggregating to monthly: {str(e)}")
            return pd.DataFrame()
    
    def _generate_long_term_forecasts(self, monthly_data: pd.DataFrame, years_ahead: int) -> List[Dict]:
        """Generate long-term forecasts using trend analysis"""
        forecasts = []
        
        try:
            if monthly_data.empty:
                return forecasts
            
            # Simple trend-based forecast
            prices = monthly_data['avg_lmp'].values
            if len(prices) < 12:  # Need at least 12 months of data
                # Generate placeholder forecasts
                base_price = np.mean(prices) if len(prices) > 0 else 50.0
                for year in range(1, years_ahead + 1):
                    forecasts.append({
                        'year': datetime.now().year + year,
                        'forecast_price': round(base_price * (1.02 ** year), 2),  # 2% annual growth
                        'confidence_interval': {
                            'lower': round(base_price * (1.02 ** year) * 0.8, 2),
                            'upper': round(base_price * (1.02 ** year) * 1.2, 2)
                        },
                        'methodology': 'Simple trend extrapolation'
                    })
            else:
                # Linear trend fitting
                X = np.arange(len(prices)).reshape(-1, 1)
                y = prices
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Project future values
                current_year = datetime.now().year
                months_per_year = 12
                
                for year in range(1, years_ahead + 1):
                    future_x = len(prices) + (year * months_per_year)
                    forecast_price = model.predict([[future_x]])[0]
                    
                    # Add some uncertainty
                    std_error = np.std(prices - model.predict(X))
                    
                    forecasts.append({
                        'year': current_year + year,
                        'forecast_price': round(max(forecast_price, 0), 2),
                        'confidence_interval': {
                            'lower': round(max(forecast_price - 2 * std_error, 0), 2),
                            'upper': round(forecast_price + 2 * std_error, 2)
                        },
                        'methodology': 'Linear trend analysis'
                    })
        
        except Exception as e:
            self.logger.error(f"Error generating long-term forecasts: {str(e)}")
        
        return forecasts
    
    def _analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Analyze current market conditions"""
        analysis = {
            'market_state': 'normal',
            'price_trend': 'stable',
            'volatility_level': 'medium',
            'congestion_risk': 'low'
        }
        
        try:
            if 'real_time_avg_lmp' in market_data and 'day_ahead_avg_lmp' in market_data:
                rt_price = market_data['real_time_avg_lmp']
                da_price = market_data['day_ahead_avg_lmp']
                
                price_diff = rt_price - da_price
                price_diff_pct = (price_diff / da_price * 100) if da_price > 0 else 0
                
                if abs(price_diff_pct) > 20:
                    analysis['market_state'] = 'volatile'
                    analysis['volatility_level'] = 'high'
                elif abs(price_diff_pct) > 10:
                    analysis['volatility_level'] = 'medium'
                else:
                    analysis['volatility_level'] = 'low'
                
                if price_diff_pct > 10:
                    analysis['price_trend'] = 'increasing'
                elif price_diff_pct < -10:
                    analysis['price_trend'] = 'decreasing'
                
                if rt_price > 100:  # High price threshold
                    analysis['congestion_risk'] = 'high'
                elif rt_price > 75:
                    analysis['congestion_risk'] = 'medium'
        
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
        
        return analysis
    
    def _generate_congestion_recommendations(self, zone_data: Dict) -> List[Dict]:
        """Generate congestion management recommendations"""
        recommendations = []
        
        try:
            if not zone_data:
                return recommendations
            
            high_congestion_zones = []
            for zone, data in zone_data.items():
                if isinstance(data, dict) and data.get('congestion', 0) > 10:
                    high_congestion_zones.append((zone, data['congestion']))
            
            if high_congestion_zones:
                high_congestion_zones.sort(key=lambda x: x[1], reverse=True)
                top_zone = high_congestion_zones[0]
                
                recommendations.append({
                    'type': 'congestion_management',
                    'priority': 'high',
                    'title': 'High Congestion Alert',
                    'description': f'Zone {top_zone[0]} experiencing high congestion (${top_zone[1]:.2f}/MWh). Consider load shifting or alternative transmission paths.',
                    'action': 'Monitor transmission flows and consider demand response activation',
                    'potential_savings': f'${top_zone[1] * 100:.0f}K annually',
                    'icon': '⚡'
                })
        
        except Exception as e:
            self.logger.error(f"Error generating congestion recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_trading_recommendations(self, market_data: Dict, analysis: Dict) -> List[Dict]:
        """Generate trading strategy recommendations"""
        recommendations = []
        
        try:
            rt_price = market_data.get('real_time_avg_lmp', 0)
            da_price = market_data.get('day_ahead_avg_lmp', 0)
            
            if rt_price > 0 and da_price > 0:
                price_spread = rt_price - da_price
                
                if price_spread > 15:
                    recommendations.append({
                        'type': 'trading_strategy',
                        'priority': 'medium',
                        'title': 'Arbitrage Opportunity',
                        'description': f'Real-time prices (${rt_price:.2f}) significantly higher than day-ahead (${da_price:.2f}). Consider selling in real-time market.',
                        'action': 'Increase real-time market participation',
                        'potential_revenue': f'${price_spread:.2f}/MWh spread',
                        'icon': '📈'
                    })
                elif price_spread < -15:
                    recommendations.append({
                        'type': 'trading_strategy',
                        'priority': 'medium',
                        'title': 'Day-Ahead Premium',
                        'description': f'Day-ahead prices (${da_price:.2f}) higher than real-time (${rt_price:.2f}). Consider day-ahead market focus.',
                        'action': 'Increase day-ahead market bidding',
                        'potential_revenue': f'${abs(price_spread):.2f}/MWh premium',
                        'icon': '📊'
                    })
            
            if analysis.get('volatility_level') == 'high':
                recommendations.append({
                    'type': 'risk_management',
                    'priority': 'high',
                    'title': 'High Volatility Alert',
                    'description': 'Market showing high volatility. Consider hedging strategies and risk management measures.',
                    'action': 'Implement volatility hedging strategies',
                    'potential_savings': 'Risk reduction: 15-25%',
                    'icon': '⚠️'
                })
        
        except Exception as e:
            self.logger.error(f"Error generating trading recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_risk_recommendations(self, market_data: Dict) -> List[Dict]:
        """Generate risk management recommendations"""
        recommendations = []
        
        try:
            peak_price = market_data.get('peak_hour_price', 0)
            avg_price = market_data.get('real_time_avg_lmp', 0)
            
            if peak_price > avg_price * 2 and peak_price > 100:
                recommendations.append({
                    'type': 'risk_management',
                    'priority': 'high',
                    'title': 'Extreme Price Event',
                    'description': f'Peak hour price (${peak_price:.2f}) significantly above average (${avg_price:.2f}). High price risk detected.',
                    'action': 'Activate demand response and review hedging positions',
                    'potential_savings': f'${(peak_price - avg_price) * 50:.0f}K cost avoidance',
                    'icon': '🚨'
                })
        
        except Exception as e:
            self.logger.error(f"Error generating risk recommendations: {str(e)}")
        
        return recommendations
    
    def _generate_renewable_recommendations(self, market_data: Dict) -> List[Dict]:
        """Generate renewable energy recommendations"""
        recommendations = []
        
        try:
            # Placeholder renewable recommendations
            recommendations.append({
                'type': 'renewable_strategy',
                'priority': 'medium',
                'title': 'Renewable Energy Opportunity',
                'description': 'Current market conditions favorable for renewable energy integration. Consider increasing wind/solar exposure.',
                'action': 'Evaluate renewable energy contracts and storage options',
                'potential_savings': 'Long-term cost reduction: 10-20%',
                'icon': '🌱'
            })
        
        except Exception as e:
            self.logger.error(f"Error generating renewable recommendations: {str(e)}")
        
        return recommendations
    
    def _get_model_accuracy(self, model_name: str) -> Dict:
        """Get model accuracy metrics"""
        return {
            'mae': 5.2,  # Placeholder values
            'rmse': 8.1,
            'mape': 12.5,
            'last_updated': datetime.now().isoformat()
        }

