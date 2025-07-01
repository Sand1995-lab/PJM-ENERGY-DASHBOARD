import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json

class MockPJMDataService:
    """Enhanced Mock service for PJM energy market data with all utility zones"""
    
    def __init__(self):
        self.base_price = 45.0
        # ALL PJM UTILITY ZONES - Complete List
        self.zones = [
            'AECO', 'AEP', 'AP', 'ATSI', 'BGE', 'COMED', 'DAY', 'DEOK', 'DOM', 'DPL', 
            'DUQ', 'EKPC', 'JCPL', 'METED', 'PECO', 'PENELEC', 'PEPCO', 'PPL', 'PSEG', 
            'RECO', 'APS', 'DLCO', 'OVEC', 'CUB', 'SUMMIT', 'ECAR', 'MAAC', 'RFC', 
            'SERC', 'MAIN', 'FRCC', 'SPP', 'ERCOT', 'WECC', 'NPCC', 'MRO', 'TRE',
            'WESTERN_HUB', 'EASTERN_HUB', 'NORTHERN_HUB', 'SOUTHERN_HUB', 'CENTRAL_HUB',
            'OHIO_HUB', 'PENN_HUB', 'VA_HUB', 'MD_HUB', 'WV_HUB', 'NJ_HUB', 'DE_HUB'
        ]
        
        # Zone characteristics for realistic pricing
        self.zone_characteristics = {
            'WESTERN_HUB': {'base_modifier': 1.2, 'volatility': 0.15},
            'EASTERN_HUB': {'base_modifier': 0.9, 'volatility': 0.10},
            'NORTHERN_HUB': {'base_modifier': 1.1, 'volatility': 0.12},
            'SOUTHERN_HUB': {'base_modifier': 0.95, 'volatility': 0.08},
            'CENTRAL_HUB': {'base_modifier': 1.0, 'volatility': 0.10},
            'COMED': {'base_modifier': 1.15, 'volatility': 0.14},
            'PSEG': {'base_modifier': 1.18, 'volatility': 0.16},
            'PECO': {'base_modifier': 1.12, 'volatility': 0.13},
            'BGE': {'base_modifier': 1.08, 'volatility': 0.11},
            'PEPCO': {'base_modifier': 1.14, 'volatility': 0.13},
            'AEP': {'base_modifier': 0.92, 'volatility': 0.09},
            'DOM': {'base_modifier': 0.96, 'volatility': 0.10},
            'DUQ': {'base_modifier': 0.88, 'volatility': 0.08},
        }
        
    def get_market_summary(self) -> Dict:
        """Get mock market summary"""
        rt_price = self.base_price + random.uniform(-10, 15)
        da_price = self.base_price + random.uniform(-8, 12)
        peak_price = max(rt_price, da_price) + random.uniform(5, 25)
        load = random.uniform(80000, 150000)
        
        return {
            'status': 'success',
            'data': {
                'real_time_avg_lmp': round(rt_price, 2),
                'day_ahead_avg_lmp': round(da_price, 2),
                'peak_hour_price': round(peak_price, 2),
                'current_load': round(load / 1000, 1),  # Convert to GW
                'market_status': 'NORMAL',
                'timestamp': datetime.now().isoformat(),
                'data_status': {
                    'real_time': 'success',
                    'day_ahead': 'success',
                    'load': 'success'
                }
            }
        }
    
    def get_real_time_lmp(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Get mock real-time LMP data"""
        if not start_date:
            start_date = datetime.now() - timedelta(hours=24)
        if not end_date:
            end_date = datetime.now()
            
        # Generate hourly data points
        data_points = []
        current_time = start_date
        current_price = self.base_price
        
        while current_time <= end_date:
            # Add some realistic price variation
            price_change = random.uniform(-5, 5)
            current_price = max(10, current_price + price_change)
            
            # Add daily patterns (higher during peak hours)
            hour = current_time.hour
            if 6 <= hour <= 22:  # Peak hours
                current_price *= random.uniform(1.1, 1.4)
            else:  # Off-peak hours
                current_price *= random.uniform(0.8, 1.0)
            
            data_points.append({
                'Time': current_time.isoformat(),
                'LMP': round(current_price, 2),
                'Energy': round(current_price * random.uniform(0.85, 0.95), 2),
                'Congestion': round(random.uniform(-2, 8), 2),
                'Loss': round(random.uniform(0.5, 3), 2),
                'Location': 'PJM_SYSTEM',
                'Location Type': 'SYSTEM'
            })
            
            current_time += timedelta(hours=1)
            
        return {
            'status': 'success',
            'data': data_points,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data_points)
        }
    
    def get_day_ahead_lmp(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Get mock day-ahead LMP data"""
        if not start_date:
            start_date = datetime.now().date()
        if not end_date:
            end_date = start_date + timedelta(days=1)
            
        data_points = []
        current_time = datetime.combine(start_date, datetime.min.time())
        current_price = self.base_price
        
        for hour in range(24):
            # Day-ahead prices are typically more stable
            price_change = random.uniform(-2, 2)
            current_price = max(10, current_price + price_change)
            
            # Peak hour adjustments
            if 6 <= hour <= 22:
                current_price *= random.uniform(1.05, 1.25)
            else:
                current_price *= random.uniform(0.9, 1.0)
            
            data_points.append({
                'Time': (current_time + timedelta(hours=hour)).isoformat(),
                'LMP': round(current_price, 2),
                'Energy': round(current_price * random.uniform(0.9, 0.98), 2),
                'Congestion': round(random.uniform(-1, 4), 2),
                'Loss': round(random.uniform(0.3, 2), 2),
                'Location': 'PJM_SYSTEM',
                'Location Type': 'SYSTEM'
            })
            
        return {
            'status': 'success',
            'data': data_points,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data_points)
        }
    
    def get_zone_wise_lmp(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Get mock zone-wise LMP data for ALL zones"""
        zone_data = {}
        
        # Generate data for ALL zones
        for zone in self.zones:
            # Get zone characteristics or use defaults
            char = self.zone_characteristics.get(zone, {'base_modifier': 1.0, 'volatility': 0.10})
            
            base_zone_price = self.base_price * char['base_modifier']
            price_variation = base_zone_price * char['volatility'] * random.uniform(-2, 2)
            final_price = base_zone_price + price_variation
            
            # Add congestion based on zone type
            if 'HUB' in zone or zone in ['COMED', 'PSEG', 'PECO']:
                congestion = random.uniform(2, 15)  # Higher congestion for major hubs
            else:
                congestion = random.uniform(-2, 8)
            
            zone_data[zone] = {
                'location': zone,
                'lmp': round(final_price, 2),
                'energy': round(final_price * random.uniform(0.85, 0.95), 2),
                'congestion': round(congestion, 2),
                'loss': round(random.uniform(0.5, 4), 2),
                'timestamp': datetime.now().isoformat()
            }
            
        return {
            'status': 'success',
            'data': zone_data,
            'timestamp': datetime.now().isoformat(),
            'zones': len(zone_data)
        }
    
    def get_5_year_forecast(self) -> Dict:
        """Get 5-year price forecast with detailed projections"""
        current_year = datetime.now().year
        forecasts = []
        
        # Base forecast with realistic growth assumptions
        base_price = self.base_price
        annual_growth_rates = [0.03, 0.045, 0.038, 0.042, 0.035]  # Varying growth rates
        
        for i, growth_rate in enumerate(annual_growth_rates):
            year = current_year + i + 1
            
            # Calculate forecast price with growth
            forecast_price = base_price * (1 + growth_rate) ** (i + 1)
            
            # Add market uncertainty bands
            uncertainty = forecast_price * (0.15 + i * 0.05)  # Increasing uncertainty over time
            
            # Seasonal variations
            seasonal_data = []
            for month in range(1, 13):
                # Summer months (Jun-Aug) typically higher
                if month in [6, 7, 8]:
                    seasonal_multiplier = random.uniform(1.1, 1.3)
                # Winter months (Dec-Feb) moderate
                elif month in [12, 1, 2]:
                    seasonal_multiplier = random.uniform(1.05, 1.15)
                # Spring/Fall months lower
                else:
                    seasonal_multiplier = random.uniform(0.9, 1.05)
                
                monthly_price = forecast_price * seasonal_multiplier
                seasonal_data.append({
                    'month': month,
                    'month_name': datetime(year, month, 1).strftime('%B'),
                    'forecast_price': round(monthly_price, 2),
                    'lower_bound': round(monthly_price - uncertainty * 0.8, 2),
                    'upper_bound': round(monthly_price + uncertainty * 0.8, 2)
                })
            
            forecasts.append({
                'year': year,
                'annual_avg_price': round(forecast_price, 2),
                'confidence_interval': {
                    'lower': round(forecast_price - uncertainty, 2),
                    'upper': round(forecast_price + uncertainty, 2)
                },
                'growth_rate': f"{growth_rate*100:.1f}%",
                'key_factors': self._get_forecast_factors(i),
                'seasonal_breakdown': seasonal_data,
                'peak_summer_price': round(max([s['forecast_price'] for s in seasonal_data]), 2),
                'low_spring_price': round(min([s['forecast_price'] for s in seasonal_data]), 2)
            })
        
        return {
            'status': 'success',
            'forecasts': forecasts,
            'methodology': 'Advanced econometric modeling with seasonal adjustments',
            'last_updated': datetime.now().isoformat(),
            'confidence_level': '85%',
            'assumptions': [
                'Steady renewable integration growth',
                'Moderate natural gas price volatility',
                'Normal weather patterns',
                'Continued grid modernization investments'
            ]
        }
    
    def _get_forecast_factors(self, year_index: int) -> List[str]:
        """Get key factors affecting forecast for each year"""
        factors_by_year = [
            ['Carbon pricing implementation', 'Grid modernization', 'Renewable integration'],
            ['Energy storage deployment', 'Transmission upgrades', 'Demand response growth'],
            ['Advanced nuclear deployment', 'Hydrogen integration', 'Grid flexibility'],
            ['Carbon capture technology', 'Vehicle electrification', 'Industrial heat pumps'],
            ['Full grid digitalization', 'Renewable curtailment solutions', 'Market coupling']
        ]
        return factors_by_year[year_index] if year_index < len(factors_by_year) else ['Market maturation']
    
    def get_load_data(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Get mock load data"""
        if not start_date:
            start_date = datetime.now() - timedelta(hours=24)
        if not end_date:
            end_date = datetime.now()
            
        data_points = []
        current_time = start_date
        base_load = 100000  # MW
        
        while current_time <= end_date:
            # Add daily load patterns
            hour = current_time.hour
            if 6 <= hour <= 22:  # Peak hours
                load_multiplier = random.uniform(1.1, 1.4)
            else:  # Off-peak hours
                load_multiplier = random.uniform(0.7, 0.9)
                
            current_load = base_load * load_multiplier + random.uniform(-5000, 5000)
            
            data_points.append({
                'Time': current_time.isoformat(),
                'Load': round(current_load, 1),
                'Location': 'PJM_SYSTEM'
            })
            
            current_time += timedelta(hours=1)
            
        return {
            'status': 'success',
            'data': data_points,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data_points)
        }
    
    def get_fuel_mix(self, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Get mock fuel mix data"""
        fuel_types = ['Natural Gas', 'Coal', 'Nuclear', 'Wind', 'Solar', 'Hydro', 'Oil', 'Other']
        
        data_points = []
        current_time = start_date or (datetime.now() - timedelta(hours=24))
        
        while current_time <= (end_date or datetime.now()):
            total_generation = random.uniform(80000, 140000)
            
            # Realistic fuel mix percentages
            fuel_mix = {
                'Natural Gas': random.uniform(0.35, 0.45),
                'Coal': random.uniform(0.15, 0.25),
                'Nuclear': random.uniform(0.25, 0.35),
                'Wind': random.uniform(0.05, 0.15),
                'Solar': random.uniform(0.02, 0.08),
                'Hydro': random.uniform(0.02, 0.06),
                'Oil': random.uniform(0.01, 0.03),
                'Other': random.uniform(0.01, 0.03)
            }
            
            # Normalize to 100%
            total_pct = sum(fuel_mix.values())
            fuel_mix = {k: v/total_pct for k, v in fuel_mix.items()}
            
            for fuel, percentage in fuel_mix.items():
                data_points.append({
                    'Time': current_time.isoformat(),
                    'Fuel': fuel,
                    'Generation': round(total_generation * percentage, 1),
                    'Percentage': round(percentage * 100, 2)
                })
                
            current_time += timedelta(hours=1)
            
        return {
            'status': 'success',
            'data': data_points,
            'timestamp': datetime.now().isoformat(),
            'data_points': len(data_points)
        }
