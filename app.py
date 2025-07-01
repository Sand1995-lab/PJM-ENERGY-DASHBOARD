import os
import sys
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import your existing services
from mock_data_service import MockPJMDataService
from ai_prediction_service import PJMAIPredictionService

# Create Flask app
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
CORS(app, origins="*")

# Initialize services
pjm_service = MockPJMDataService()
ai_service = PJMAIPredictionService()

# Health check
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'PJM Dashboard', 'version': '2.0'})

# Enhanced API Routes
@app.route('/api/pjm/market-summary')
def market_summary():
    return jsonify(pjm_service.get_market_summary())

@app.route('/api/pjm/real-time-lmp')
def real_time_lmp():
    hours_back = request.args.get('hours_back', 24, type=int)
    from datetime import datetime, timedelta
    start_date = datetime.now() - timedelta(hours=hours_back)
    end_date = datetime.now()
    return jsonify(pjm_service.get_real_time_lmp(start_date, end_date))

@app.route('/api/pjm/day-ahead-lmp')
def day_ahead_lmp():
    return jsonify(pjm_service.get_day_ahead_lmp())

@app.route('/api/pjm/zone-wise-lmp')
def zone_wise_lmp():
    """Get ALL utility zones LMP data"""
    return jsonify(pjm_service.get_zone_wise_lmp())

@app.route('/api/pjm/all-zones')
def all_zones():
    """Get comprehensive data for all PJM utility zones"""
    zone_data = pjm_service.get_zone_wise_lmp()
    if zone_data['status'] == 'success':
        # Sort zones by LMP price for better display
        sorted_zones = sorted(
            zone_data['data'].items(), 
            key=lambda x: x[1]['lmp'], 
            reverse=True
        )
        
        return jsonify({
            'status': 'success',
            'total_zones': len(sorted_zones),
            'zones': dict(sorted_zones),
            'highest_price_zone': sorted_zones[0] if sorted_zones else None,
            'lowest_price_zone': sorted_zones[-1] if sorted_zones else None,
            'average_lmp': round(sum(zone[1]['lmp'] for zone in sorted_zones) / len(sorted_zones), 2) if sorted_zones else 0,
            'timestamp': zone_data['timestamp']
        })
    return zone_data

@app.route('/api/pjm/5-year-forecast')
def five_year_forecast():
    """Get detailed 5-year price forecast"""
    return jsonify(pjm_service.get_5_year_forecast())

@app.route('/api/pjm/load-data')
def load_data():
    return jsonify(pjm_service.get_load_data())

@app.route('/api/pjm/fuel-mix')
def fuel_mix():
    return jsonify(pjm_service.get_fuel_mix())

@app.route('/api/pjm/predict-lmp')
def predict_lmp():
    hours_ahead = request.args.get('hours_ahead', 24, type=int)
    historical = pjm_service.get_real_time_lmp()
    if historical['status'] == 'success':
        return jsonify(ai_service.predict_short_term_lmp(historical['data'], hours_ahead))
    return jsonify({'status': 'error', 'message': 'No data'})

@app.route('/api/pjm/ai-recommendations')
def ai_recommendations():
    market_data = pjm_service.get_market_summary()
    zone_data = pjm_service.get_zone_wise_lmp()
    if market_data['status'] == 'success':
        return jsonify(ai_service.generate_ai_recommendations(
            market_data['data'], zone_data.get('data', {})))
    return jsonify({'status': 'error', 'message': 'No data'})

@app.route('/api/pjm/long-term-forecast')
def long_term_forecast():
    """Enhanced long-term forecast using AI service"""
    historical = pjm_service.get_real_time_lmp()
    if historical['status'] == 'success':
        ai_forecast = ai_service.predict_long_term_forecast(historical['data'])
        # Combine with detailed 5-year forecast
        detailed_forecast = pjm_service.get_5_year_forecast()
        
        return jsonify({
            'status': 'success',
            'ai_forecast': ai_forecast.get('forecasts', []),
            'detailed_forecast': detailed_forecast.get('forecasts', []),
            'methodology': 'Combined AI and econometric modeling',
            'timestamp': detailed_forecast.get('last_updated')
        })
    return jsonify({'status': 'error', 'message': 'No data'})

@app.route('/api/pjm/risk-metrics')
def risk_metrics():
    historical = pjm_service.get_real_time_lmp()
    if historical['status'] == 'success':
        return jsonify(ai_service.calculate_risk_metrics(historical['data']))
    return jsonify({'status': 'error', 'message': 'No data'})

@app.route('/api/pjm/zones/top-congested')
def top_congested_zones():
    """Get top 10 most congested zones"""
    zone_data = pjm_service.get_zone_wise_lmp()
    if zone_data['status'] == 'success':
        zones = zone_data['data']
        # Sort by congestion
        congested = sorted(
            zones.items(),
            key=lambda x: x[1]['congestion'],
            reverse=True
        )[:10]
        
        return jsonify({
            'status': 'success',
            'top_congested_zones': [
                {
                    'zone': zone,
                    'congestion': data['congestion'],
                    'lmp': data['lmp'],
                    'energy': data['energy']
                }
                for zone, data in congested
            ]
        })
    return jsonify({'status': 'error', 'message': 'No zone data'})

@app.route('/api/pjm/zones/price-ranking')
def zones_price_ranking():
    """Get zones ranked by price"""
    zone_data = pjm_service.get_zone_wise_lmp()
    if zone_data['status'] == 'success':
        zones = zone_data['data']
        ranked = sorted(zones.items(), key=lambda x: x[1]['lmp'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'price_ranking': [
                {
                    'rank': i + 1,
                    'zone': zone,
                    'lmp': data['lmp'],
                    'congestion': data['congestion'],
                    'loss': data['loss']
                }
                for i, (zone, data) in enumerate(ranked)
            ],
            'total_zones': len(ranked)
        })
    return jsonify({'status': 'error', 'message': 'No zone data'})

# API documentation endpoint
@app.route('/api')
def api_docs():
    return jsonify({
        'service': 'PJM Advanced Energy Analytics API',
        'version': '2.0',
        'features': ['5-Year Forecasting', 'All Utility Zones', 'AI Predictions'],
        'endpoints': {
            'market_summary': '/api/pjm/market-summary',
            'all_zones': '/api/pjm/all-zones',
            'five_year_forecast': '/api/pjm/5-year-forecast',
            'zone_price_ranking': '/api/pjm/zones/price-ranking',
            'top_congested_zones': '/api/pjm/zones/top-congested',
            'real_time_lmp': '/api/pjm/real-time-lmp',
            'day_ahead_lmp': '/api/pjm/day-ahead-lmp',
            'load_data': '/api/pjm/load-data',
            'fuel_mix': '/api/pjm/fuel-mix',
            'ai_predictions': '/api/pjm/predict-lmp',
            'ai_recommendations': '/api/pjm/ai-recommendations',
            'long_term_forecast': '/api/pjm/long-term-forecast',
            'risk_metrics': '/api/pjm/risk-metrics'
        },
        'total_zones': len(pjm_service.zones),
        'status': 'operational'
    })

# Serve static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path == "" or path == "index.html":
        if os.path.exists('index.html'):
            return send_from_directory('.', 'index.html')
    
    if os.path.exists(path):
        return send_from_directory('.', path)
    
    return jsonify({
        'message': 'PJM Dashboard API v2.0', 
        'features': ['5-Year Forecasts', 'All Utility Zones'],
        'api_docs': '/api'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
