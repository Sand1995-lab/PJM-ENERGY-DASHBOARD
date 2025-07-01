import os
import sys
from flask import Flask, send_from_directory, jsonify
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
    return jsonify({'status': 'healthy', 'service': 'PJM Dashboard'})

# API Routes
@app.route('/api/pjm/market-summary')
def market_summary():
    return jsonify(pjm_service.get_market_summary())

@app.route('/api/pjm/real-time-lmp')
def real_time_lmp():
    return jsonify(pjm_service.get_real_time_lmp())

@app.route('/api/pjm/day-ahead-lmp')
def day_ahead_lmp():
    return jsonify(pjm_service.get_day_ahead_lmp())

@app.route('/api/pjm/zone-wise-lmp')
def zone_wise_lmp():
    return jsonify(pjm_service.get_zone_wise_lmp())

@app.route('/api/pjm/load-data')
def load_data():
    return jsonify(pjm_service.get_load_data())

@app.route('/api/pjm/fuel-mix')
def fuel_mix():
    return jsonify(pjm_service.get_fuel_mix())

@app.route('/api/pjm/predict-lmp')
def predict_lmp():
    historical = pjm_service.get_real_time_lmp()
    if historical['status'] == 'success':
        return jsonify(ai_service.predict_short_term_lmp(historical['data']))
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
    historical = pjm_service.get_real_time_lmp()
    if historical['status'] == 'success':
        return jsonify(ai_service.predict_long_term_forecast(historical['data']))
    return jsonify({'status': 'error', 'message': 'No data'})

@app.route('/api/pjm/risk-metrics')
def risk_metrics():
    historical = pjm_service.get_real_time_lmp()
    if historical['status'] == 'success':
        return jsonify(ai_service.calculate_risk_metrics(historical['data']))
    return jsonify({'status': 'error', 'message': 'No data'})

# Serve static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path == "" or path == "index.html":
        if os.path.exists('index.html'):
            return send_from_directory('.', 'index.html')
    
    if os.path.exists(path):
        return send_from_directory('.', path)
    
    return jsonify({'message': 'PJM Dashboard API', 'health': '/health'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
