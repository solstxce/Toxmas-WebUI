from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, make_response
from flask_cors import CORS
import requests
import os
from functools import wraps
from datetime import datetime, timedelta

app = Flask(__name__, template_folder="templates")
CORS(app)

# Configuration
BACKEND_URL = "http://localhost:5000"  # Your backend URL
app.config['SECRET_KEY'] = 'your_frontend_secret_key'  # Change this to a secure random key

# Add these constants at the top of the file
UMAMI_API_URL = "https://api.umami.is/v1"
UMAMI_WEBSITE_ID = "fd4b2aa0-cdf8-4735-a1d5-8918dc88a047"
UMAMI_API_KEY = "VkRkplgSWFRrLp05ZIRoH5pqtSIjVOZh"

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    # token = request.cookies.get('token')
    # if token:
    #     return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/home')
@login_required
def home():
    token = request.cookies.get('token')
    headers = {'Authorization': token}
    
    recent_analyses = requests.get(f"{BACKEND_URL}/recent_analyses", headers=headers).json()
    content_types = requests.get(f"{BACKEND_URL}/content_types", headers=headers).json()
    
    return render_template('home.html', recent_analyses=recent_analyses, content_types=content_types)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        response = requests.post(f"{BACKEND_URL}/login", json=data)
        if response.status_code == 200:
            token = response.json().get('token')
            print(token)
            resp = make_response(jsonify({'message': 'Login successful', 'token': token, 'redirect': url_for('home')}))
            resp.set_cookie('token', token, httponly=True)
            return resp, 200
        return jsonify(response.json()), response.status_code
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        response = requests.post(f"{BACKEND_URL}/register", json=data)
        return jsonify(response.json()), response.status_code
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

# @app.route('/dashboard/scan')
# @login_required
# def dashboard_scan():
#     return render_template('dashboard/scan.html')

# @app.route('/dashboard/history')
# @login_required
# def dashboard_history():
#     token = request.cookies.get('token')
#     headers = {'Authorization': token}
#     past_analyses = requests.get(f"{BACKEND_URL}/past_analyses", headers=headers).json()
#     return render_template('dashboard/history.html', past_analyses=past_analyses)

# @app.route('/dashboard/profile')
# @login_required
# def dashboard_profile():
#     token = request.cookies.get('token')
#     headers = {'Authorization': token}
#     user_data = requests.get(f"{BACKEND_URL}/user_profile", headers=headers).json()
#     return render_template('dashboard/profile.html', user_data=user_data)

# # Add these new API endpoints for dashboard data
# @app.route('/api/dashboard/stats')
# @login_required
# def dashboard_stats():
#     token = request.cookies.get('token')
#     headers = {'Authorization': token}
    
#     # Fetch all required dashboard data
#     recent_analyses = requests.get(f"{BACKEND_URL}/recent_analyses", headers=headers).json()
#     content_types = requests.get(f"{BACKEND_URL}/content_types", headers=headers).json()
#     page_visits = requests.get(f"{BACKEND_URL}/api/page_visits", headers=headers).json()
#     user_trend = requests.get(f"{BACKEND_URL}/api/user_trend", headers=headers).json()
    
#     return jsonify({
#         'recent_analyses': recent_analyses,
#         'content_types': content_types,
#         'page_visits': page_visits,
#         'user_trend': user_trend
#     })

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    token = request.cookies.get('token')
    if not token:
        return jsonify({'message': 'Token is missing'}), 401

    file = request.files.get('file')
    if not file:
        return jsonify({'message': 'No file part'}), 400

    files = {'file': (file.filename, file.stream, file.content_type)}
    headers = {'Authorization': token}

    response = requests.post(f"{BACKEND_URL}/analyze", files=files, headers=headers)

    if response.headers.get('Content-Type') == 'application/json':
        return jsonify(response.json()), response.status_code
    elif response.headers.get('Content-Type') in ['application/pdf', 'video/mp4']:
        return send_file(
            response.content,
            mimetype=response.headers.get('Content-Type'),
            as_attachment=True,
            download_name=response.headers.get('Content-Disposition', 'attachment').split('filename=')[-1].strip('"')
        )
    else:
        return response.content, response.status_code, response.headers.items()

@app.route('/static/<path:path>')
def send_static(path):
    return send_file(f'static/{path}')

@app.route('/logout')
def logout():
    resp = make_response(redirect(url_for('login')))
    resp.set_cookie('token', '', expires=0)
    flash('You have been logged out.', 'info')
    return resp

@app.route('/api/user_trend')
@login_required
def get_user_trend():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    headers = {
        'x-umami-api-key': UMAMI_API_KEY
    }
    
    response = requests.get(
        f"{UMAMI_API_URL}/websites/{UMAMI_WEBSITE_ID}/metrics",
        params={
            'startAt': int(start_date.timestamp() * 1000),
            'endAt': int(end_date.timestamp() * 1000),
            'type': 'browser'
        },
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        # user_trend = [{'date': datetime.fromtimestamp(item['x'] / 1000).strftime('%Y-%m-%d'), 'visitors': item['y']} for item in data]
        return jsonify(data)
    else:
        return jsonify({'error': 'Failed to fetch user trend data'}), 500

@app.route('/api/page_visits')
@login_required
def get_page_visits():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    headers = {
        'x-umami-api-key': UMAMI_API_KEY
    }
    
    response = requests.get(
        f"{UMAMI_API_URL}/websites/{UMAMI_WEBSITE_ID}/stats",
        params={
            'startAt': int(start_date.timestamp() * 1000),
            'endAt': int(end_date.timestamp() * 1000)
        },
        headers=headers
    )
    
    if response.status_code == 200:
        data = response.json()
        return jsonify({'pageVisits': data['pageviews']['value']})
    else:
        return jsonify({'error': 'Failed to fetch page visits data'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Run on a different port than the backend
