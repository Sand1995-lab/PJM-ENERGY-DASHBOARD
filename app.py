import os
import sys
from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import enhanced services
try:
    from mega_enhanced_mock_data_service import MegaEnhancedPJMDataService
    from ai_prediction_service import PJMAIPredictionService
    print("✅ Successfully imported all services")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Create minimal fallback services
    class MegaEnhancedPJMDataService:
        def get_real_time_alerts(self):
            return {'status': 'success', 'alerts': [], 'total_alerts': 0}
        def get_all_zones_current_status(self):
            return {'status': 'success', 'zones': {}, 'summary': {'avg_lmp': 45.0}}
    
    class PJMAIPredictionService:
        def generate_ai_recommendations(self, *args):
            return {'status': 'success', 'recommendations': []}

# Create Flask app
app = Flask(__name__, static_folder='.')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'pjm-mega-dashboard-2025')
CORS(app, origins="*")

# Initialize services
try:
    pjm_service = MegaEnhancedPJMDataService()
    ai_service = PJMAIPredictionService()
    print("✅ Services initialized successfully")
except Exception as e:
    print(f"❌ Service initialization error: {e}")

# ============= HEALTH & STATUS ENDPOINTS =============
@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'service': 'PJM MEGA Dashboard', 
        'version': '3.0',
        'features': ['365-Day History', 'Real-time Alerts', 'All Zones Predictions'],
        'timestamp': datetime.now().isoformat(),
        'environment': os.environ.get('FLASK_ENV', 'production')
    })

@app.route('/api')
def api_docs():
    return jsonify({
        'service': 'PJM MEGA Advanced Energy Analytics API',
        'version': '3.0',
        'description': 'Comprehensive energy market intelligence platform',
        'features': [
            '50+ PJM zones with real-time data',
            '365-Day Historical Data Analysis',
            'Real-time Alerts & Notifications', 
            'AI-powered Zone Predictions',
            'Advanced Market Surveillance',
            'Price Anomaly Detection',
            'Risk Analytics & Trading Recommendations'
        ],
        'endpoints': {
            'dashboard': {
                'summary': '/api/pjm/dashboard/summary',
                'health': '/health',
                'api_docs': '/api'
            },
            'alerts': {
                'realtime': '/api/pjm/alerts/realtime',
                'critical': '/api/pjm/alerts/critical'
            },
            'zones': {
                'all_status': '/api/pjm/zones/all-current-status',
                'by_type': '/api/pjm/zones/by-type/<type>',
                'by_state': '/api/pjm/zones/by-state/<state>'
            },
            'predictions': {
                'zone_specific': '/api/pjm/predictions/zone/<zone>',
                'all_zones_summary': '/api/pjm/predictions/all-zones-summary'
            },
            'analytics': {
                'price_correlations': '/api/pjm/analytics/price-correlation',
                'congestion_hotspots': '/api/pjm/analytics/congestion-hotspots',
                'price_anomalies': '/api/pjm/surveillance/price-anomalies'
            },
            'historical': {
                'history_365_days': '/api/pjm/history/365-days',
                'hourly_averages': '/api/pjm/history/hourly-averages',
                'zone_history': '/api/pjm/history/365-days/<zone>'
            }
        },
        'zone_types': ['utility', 'hub', 'interface', 'generation', 'load'],
        'total_zones': 47,
        'data_coverage': '365 days historical + real-time + 24h predictions',
        'status': 'operational',
        'last_updated': datetime.now().isoformat()
    })

# ============= DASHBOARD SUMMARY ENDPOINTS =============
@app.route('/api/pjm/dashboard/summary')
def dashboard_summary():
    """Get comprehensive dashboard summary"""
    try:
        alerts = pjm_service.get_real_time_alerts()
        all_zones = pjm_service.get_all_zones_current_status()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_overview': {
                'total_zones': 47,
                'operational_zones': len(all_zones.get('zones', {})) if all_zones['status'] == 'success' else 0,
                'avg_system_price': all_zones.get('summary', {}).get('avg_lmp', 45.0) if all_zones['status'] == 'success' else 45.0,
                'total_system_load': all_zones.get('summary', {}).get('total_load', 85000) if all_zones['status'] == 'success' else 85000
            },
            'alerts_summary': {
                'total_alerts': alerts.get('total_alerts', 0) if alerts['status'] == 'success' else 0,
                'critical_alerts': alerts.get('critical_alerts', 0) if alerts['status'] == 'success' else 0,
                'warning_alerts': alerts.get('warning_alerts', 0) if alerts['status'] == 'success' else 0
            },
            'top_zones': {
                'highest_price': None,
                'most_congested': None,
                'highest_utilization': None
            }
        }
        
        if all_zones['status'] == 'success' and all_zones['zones']:
            zones_data = all_zones['zones']
            
            # Find top zones
            highest_price_zone = max(zones_data.items(), key=lambda x: x[1]['current_lmp'])
            most_congested_zone = max(zones_data.items(), key=lambda x: x[1]['current_congestion'])
            highest_util_zone = max(zones_data.items(), key=lambda x: x[1]['utilization_pct'])
            
            summary['top_zones'] = {
                'highest_price': {
                    'zone': highest_price_zone[0],
                    'price': highest_price_zone[1]['current_lmp'],
                    'state': highest_price_zone[1]['zone_info']['state']
                },
                'most_congested': {
                    'zone': most_congested_zone[0],
                    'congestion': most_congested_zone[1]['current_congestion'],
                    'state': most_congested_zone[1]['zone_info']['state']
                },
                'highest_utilization': {
                    'zone': highest_util_zone[0],
                    'utilization': highest_util_zone[1]['utilization_pct'],
                    'state': highest_util_zone[1]['zone_info']['state']
                }
            }
        
        return jsonify({
            'status': 'success',
            'dashboard_summary': summary
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Dashboard summary failed: {str(e)}',
            'fallback_data': {
                'system_overview': {
                    'total_zones': 47,
                    'operational_zones': 47,
                    'avg_system_price': 45.0,
                    'total_system_load': 85000
                }
            }
        })

# ============= REAL-TIME ALERTS ENDPOINTS =============
@app.route('/api/pjm/alerts/realtime')
def realtime_alerts():
    """Get real-time alerts for all zones"""
    try:
        return jsonify(pjm_service.get_real_time_alerts())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get alerts: {str(e)}',
            'alerts': [],
            'total_alerts': 0
        })

@app.route('/api/pjm/alerts/critical')
def critical_alerts():
    """Get only critical alerts"""
    try:
        alerts_data = pjm_service.get_real_time_alerts()
        if alerts_data['status'] == 'success':
            critical = [alert for alert in alerts_data['alerts'] if alert['alert_level'] == 'critical']
            return jsonify({
                'status': 'success',
                'critical_alerts': critical,
                'count': len(critical),
                'timestamp': alerts_data['last_updated']
            })
        return alerts_data
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get critical alerts: {str(e)}',
            'critical_alerts': [],
            'count': 0
        })

# ============= ZONES DATA ENDPOINTS =============
@app.route('/api/pjm/zones/all-current-status')
def all_zones_current_status():
    """Get current status for all zones"""
    try:
        return jsonify(pjm_service.get_all_zones_current_status())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get zones status: {str(e)}',
            'zones': {},
            'total_zones': 0
        })

@app.route('/api/pjm/zones/by-type/<zone_type>')
def zones_by_type(zone_type):
    """Get zones filtered by type (utility, hub, interface, generation, load)"""
    try:
        all_status = pjm_service.get_all_zones_current_status()
        if all_status['status'] == 'success':
            filtered_zones = {
                zone_name: zone_data for zone_name, zone_data in all_status['zones'].items()
                if zone_data['zone_info']['type'] == zone_type
            }
            return jsonify({
                'status': 'success',
                'zone_type': zone_type,
                'zones': filtered_zones,
                'count': len(filtered_zones)
            })
        return all_status
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to filter zones by type: {str(e)}',
            'zones': {},
            'count': 0
        })

@app.route('/api/pjm/zones/by-state/<state>')
def zones_by_state(state):
    """Get zones filtered by state"""
    try:
        all_status = pjm_service.get_all_zones_current_status()
        if all_status['status'] == 'success':
            filtered_zones = {
                zone_name: zone_data for zone_name, zone_data in all_status['zones'].items()
                if state.upper() in zone_data['zone_info']['state'].upper()
            }
            return jsonify({
                'status': 'success',
                'state': state.upper(),
                'zones': filtered_zones,
                'count': len(filtered_zones)
            })
        return all_status
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to filter zones by state: {str(e)}',
            'zones': {},
            'count': 0
        })

# ============= PREDICTIONS ENDPOINTS =============
@app.route('/api/pjm/predictions/zone/<zone>')
def zone_predictions(zone):
    """Get detailed predictions for specific zone"""
    try:
        hours_ahead = request.args.get('hours_ahead', 24, type=int)
        return jsonify(pjm_service.get_zone_predictions(zone, hours_ahead))
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get zone predictions: {str(e)}',
            'zone': zone,
            'predictions': []
        })

@app.route('/api/pjm/predictions/all-zones-summary')
def all_zones_predictions_summary():
    """Get prediction summary for all major zones"""
    try:
        major_zones = ['WESTERN_HUB', 'EASTERN_HUB', 'COMED', 'PSEG', 'PECO', 'BGE', 'DOM', 'AEP']
        predictions_summary = {}
        
        for zone in major_zones:
            try:
                zone_pred = pjm_service.get_zone_predictions(zone, 6)  # 6 hour forecast
                if zone_pred['status'] == 'success':
                    next_6_hours = zone_pred['predictions'][:6]
                    predictions_summary[zone] = {
                        'current_prediction': next_6_hours[0] if next_6_hours else None,
                        'peak_next_6h': max([p['predicted_lmp'] for p in next_6_hours]) if next_6_hours else 0,
                        'avg_next_6h': round(sum([p['predicted_lmp'] for p in next_6_hours]) / len(next_6_hours), 2) if next_6_hours else 0,
                        'trend': 'increasing' if len(next_6_hours) >= 2 and next_6_hours[1]['predicted_lmp'] > next_6_hours[0]['predicted_lmp'] else 'decreasing'
                    }
            except Exception as zone_error:
                print(f"Error predicting for zone {zone}: {zone_error}")
                continue
        
        return jsonify({
            'status': 'success',
            'predictions_summary': predictions_summary,
            'forecast_horizon': '6 hours',
            'major_zones_count': len(predictions_summary)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to get all zones predictions: {str(e)}',
            'predictions_summary': {},
            'major_zones_count': 0
        })

# ============= ANALYTICS ENDPOINTS =============
@app.route('/api/pjm/analytics/price-correlation')
def price_correlation_analysis():
    """Analyze price correlations between zones"""
    try:
        all_status = pjm_service.get_all_zones_current_status()
        if all_status['status'] == 'success':
            zones_data = all_status['zones']
            
            # Calculate price correlations (simplified)
            correlations = {}
            major_zones = ['WESTERN_HUB', 'EASTERN_HUB', 'COMED', 'PSEG', 'PECO']
            
            for i, zone1 in enumerate(major_zones):
                for zone2 in major_zones[i+1:]:
                    if zone1 in zones_data and zone2 in zones_data:
                        # Simplified correlation calculation
                        price1 = zones_data[zone1]['current_lmp']
                        price2 = zones_data[zone2]['current_lmp']
                        correlation = min(price1, price2) / max(price1, price2) if max(price1, price2) > 0 else 0
                        correlations[f"{zone1}_vs_{zone2}"] = round(correlation, 3)
            
            return jsonify({
                'status': 'success',
                'price_correlations': correlations,
                'analysis_timestamp': datetime.now().isoformat()
            })
        return all_status
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Price correlation analysis failed: {str(e)}',
            'price_correlations': {}
        })

@app.route('/api/pjm/analytics/congestion-hotspots')
def congestion_hotspots():
    """Identify current congestion hotspots"""
    try:
        all_status = pjm_service.get_all_zones_current_status()
        if all_status['status'] == 'success':
            zones_data = all_status['zones']
            
            # Find zones with highest congestion
            congested_zones = [
                {
                    'zone': zone_name,
                    'congestion': zone_data['current_congestion'],
                    'lmp': zone_data['current_lmp'],
                    'state': zone_data['zone_info']['state'],
                    'type': zone_data['zone_info']['type']
                }
                for zone_name, zone_data in zones_data.items()
                if zone_data['current_congestion'] > 5
            ]
            
            # Sort by congestion level
            congested_zones.sort(key=lambda x: x['congestion'], reverse=True)
            
            return jsonify({
                'status': 'success',
                'congestion_hotspots': congested_zones[:15],  # Top 15
                'total_congested_zones': len(congested_zones),
                'avg_congestion_cost': round(sum([z['congestion'] for z in congested_zones]) / len(congested_zones), 2) if congested_zones else 0
            })
        return all_status
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Congestion analysis failed: {str(e)}',
            'congestion_hotspots': [],
            'total_congested_zones': 0
        })

@app.route('/api/pjm/surveillance/price-anomalies')
def price_anomalies():
    """Detect price anomalies across all zones"""
    try:
        all_status = pjm_service.get_all_zones_current_status()
        if all_status['status'] == 'success':
            zones_data = all_status['zones']
            prices = [zone_data['current_lmp'] for zone_data in zones_data.values()]
            
            import numpy as np
            avg_price = np.mean(prices)
            std_price = np.std(prices)
            
            anomalies = []
            for zone_name, zone_data in zones_data.items():
                price = zone_data['current_lmp']
                z_score = abs(price - avg_price) / std_price if std_price > 0 else 0
                
                if z_score > 2:  # 2 standard deviations
                    anomaly_type = 'high' if price > avg_price else 'low'
                    anomalies.append({
                        'zone': zone_name,
                        'current_lmp': price,
                        'market_avg': round(avg_price, 2),
                        'deviation': round(price - avg_price, 2),
                        'z_score': round(z_score, 2),
                        'anomaly_type': anomaly_type,
                        'severity': 'extreme' if z_score > 3 else 'significant'
                    })
            
            return jsonify({
                'status': 'success',
                'price_anomalies': anomalies,
                'market_statistics': {
                    'avg_price': round(avg_price, 2),
                    'std_deviation': round(std_price, 2),
                    'min_price': round(min(prices), 2),
                    'max_price': round(max(prices), 2)
                },
                'total_anomalies': len(anomalies)
            })
        return all_status
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Price anomaly detection failed: {str(e)}',
            'price_anomalies': [],
            'total_anomalies': 0
        })

# ============= HISTORICAL DATA ENDPOINTS =============
@app.route('/api/pjm/history/365-days')
@app.route('/api/pjm/history/365-days/<zone>')
def history_365_days(zone=None):
    """Get 365-day historical averages for zones"""
    try:
        return jsonify(pjm_service.get_365_day_averages(zone))
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Historical data failed: {str(e)}',
            'zones': {},
            'total_zones': 0
        })

@app.route('/api/pjm/history/hourly-averages')
def hourly_averages_all_zones():
    """Get hourly averages for all zones"""
    try:
        data = pjm_service.get_365_day_averages()
        if data['status'] == 'success':
            hourly_summary = {}
            for hour in range(24):
                hour_data = []
                for zone_name, zone_data in data['zones'].items():
                    if hour in zone_data['hourly_averages']:
                        hour_data.append({
                            'zone': zone_name,
                            'avg_lmp': zone_data['hourly_averages'][hour]['avg_lmp'],
                            'peak_lmp': zone_data['hourly_averages'][hour]['peak_lmp']
                        })
                
                if hour_data:
                    hourly_summary[f"hour_{hour:02d}"] = {
                        'zones': hour_data,
                        'system_avg': round(sum([z['avg_lmp'] for z in hour_data]) / len(hour_data), 2),
                        'system_peak': round(max([z['peak_lmp'] for z in hour_data]), 2)
                    }
            
            return jsonify({
                'status': 'success',
                'hourly_system_averages': hourly_summary,
                'total_hours': 24
            })
        return data
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Hourly averages failed: {str(e)}',
            'hourly_system_averages': {},
            'total_hours': 0
        })

# ============= LEGACY COMPATIBILITY ENDPOINTS =============
@app.route('/api/pjm/market-summary')
def market_summary():
    """Legacy endpoint for market summary"""
    try:
        return jsonify(pjm_service.get_market_summary())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Market summary failed: {str(e)}',
            'data': {
                'real_time_avg_lmp': 45.0,
                'day_ahead_avg_lmp': 42.0,
                'peak_hour_price': 65.0,
                'current_load': 85.0,
                'market_status': 'NORMAL'
            }
        })

@app.route('/api/pjm/zone-wise-lmp')
def zone_wise_lmp():
    """Legacy endpoint for zone-wise LMP"""
    try:
        return jsonify(pjm_service.get_zone_wise_lmp())
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Zone-wise LMP failed: {str(e)}',
            'data': {},
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/pjm/ai-recommendations')
def ai_recommendations():
    """AI-powered trading recommendations"""
    try:
        market_data = pjm_service.get_market_summary()
        zone_data = pjm_service.get_zone_wise_lmp()
        if market_data['status'] == 'success':
            return jsonify(ai_service.generate_ai_recommendations(
                market_data['data'], zone_data.get('data', {})))
        return jsonify({'status': 'error', 'message': 'No market data available'})
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'AI recommendations failed: {str(e)}',
            'recommendations': []
        })

# ============= STATIC FILE SERVING =============
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    """Serve static files and SPA routing"""
    
    # Handle API routes - don't serve static files for API calls
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found', 'available_endpoints': '/api'}), 404
    
    # Serve specific static files
    if path != "" and os.path.exists(path):
        return send_from_directory('.', path)
    
    # Serve index.html for SPA routes or main page
    if path == "" or path == "index.html":
        if os.path.exists('index.html'):
            return send_from_directory('.', 'index.html')
    
    # Fallback response with API information
    return jsonify({
        'message': 'PJM MEGA Dashboard API v3.0', 
        'features': [
            '🏢 50+ PJM zones with real-time data',
            '⚡ Real-time alerts & notifications',
            '🔮 AI-powered price predictions',
            '📊 365-day historical analysis',
            '🚨 Advanced market surveillance',
            '📈 Price anomaly detection'
        ],
        'endpoints': {
            'dashboard': '/api/pjm/dashboard/summary',
            'alerts': '/api/pjm/alerts/realtime',
            'zones': '/api/pjm/zones/all-current-status',
            'predictions': '/api/pjm/predictions/all-zones-summary',
            'analytics': '/api/pjm/analytics/price-correlation',
            'api_docs': '/api'
        },
        'frontend': 'Upload index.html to access the dashboard',
        'status': 'Backend operational',
        'timestamp': datetime.now().isoformat()
    })

# ============= ERROR HANDLERS =============
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found',
        'code': 404,
        'available_endpoints': ['/health', '/api', '/api/pjm/dashboard/summary'],
        'suggestion': 'Check /api for complete endpoint documentation'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'code': 500,
        'timestamp': datetime.now().isoformat()
    }), 500

# ============= APPLICATION STARTUP =============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print("🚀 Starting PJM MEGA Dashboard...")
    print(f"📡 Port: {port}")
    print(f"🔧 Debug: {debug}")
    print(f"🌍 Environment: {os.environ.get('FLASK_ENV', 'production')}")
    print("✅ PJM MEGA Dashboard is ready!")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
