import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json

class MockPJMDataService:
    """Mock service for PJM energy market data for demonstration purposes"""
    
    def __init__(self):
        self.base_price = 45.0
        self.zones = ['AECO', 'AEP', 'AP', 'ATSI', 'BGE', 'COMED', 'DAY', 'DEOK', 'DOM', 'DPL', 'DUQ', 'EKPC', 'JCPL', 'METED', 'PECO', 'PENELEC', 'PEPCO', 'PPL', 'PSEG', 'RECO']
        
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
        """Get mock zone-wise LMP data"""
        zone_data = {}
        
        for zone in self.zones[:10]:  # Limit to 10 zones for demo
            base_zone_price = self.base_price + random.uniform(-15, 15)
            zone_data[zone] = {
                'location': zone,
                'lmp': round(base_zone_price, 2),
                'energy': round(base_zone_price * random.uniform(0.85, 0.95), 2),
                'congestion': round(random.uniform(-5, 15), 2),
                'loss': round(random.uniform(0.5, 4), 2),
                'timestamp': datetime.now().isoformat()
            }
            
        return {
            'status': 'success',
            'data': zone_data,
            'timestamp': datetime.now().isoformat(),
            'zones': len(zone_data)
        }
    
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

