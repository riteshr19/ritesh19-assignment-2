from flask import Flask, render_template, jsonify, request
import numpy as np
from kmeans import KMeans

app = Flask(__name__)

data = None
kmeans_model = None
manual_centroids = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    global data, kmeans_model, manual_centroids
    try:
        n_samples = int(request.form['n_samples'])
        if n_samples <= 0:
            return jsonify({'error': 'Number of samples must be positive'}), 400
        data = np.random.uniform(-10, 10, size=(n_samples, 2))
        manual_centroids = []  # Reset for manual centroids
        return jsonify(data.tolist())
    except ValueError:
        return jsonify({'error': 'Invalid input for number of samples'}), 400

@app.route('/select_manual_centroid', methods=['POST'])
def select_manual_centroid():
    global manual_centroids
    x, y = request.json['x'], request.json['y']
    manual_centroids.append([x, y])
    return jsonify(manual_centroids)

@app.route('/initialize_kmeans', methods=['POST'])
def initialize_kmeans():
    global data, kmeans_model, manual_centroids
    try:
        k = int(request.form['k'])
        method = request.form['method']
        
        # Debugging print to check received method
        print(f"Received method: {method}")

        # Ensure 'kmeans++' matches the backend expected value
        if k <= 0:
            return jsonify({'error': 'K must be positive'}), 400
        if method not in ['random', 'farthest_first', 'kmeanso', 'manual']:
            return jsonify({'error': 'Invalid initialization method'}), 400
        if method == 'manual' and len(manual_centroids) != k:
            return jsonify({'error': f'Number of manual centroids ({len(manual_centroids)}) does not match K ({k})'}), 400
        if data is None:
            return jsonify({'error': 'No data generated yet'}), 400
        
        kmeans_model = KMeans(k=k)
        kmeans_model.initialize_centroids(data, method=method, manual_centroids=manual_centroids)
        return jsonify({'centroids': kmeans_model.centroids.tolist()})
    except ValueError:
        return jsonify({'error': 'Invalid input for K'}), 400

@app.route('/step_kmeans', methods=['POST'])
def step_kmeans():
    global data, kmeans_model
    if kmeans_model is None:
        return jsonify({'error': 'KMeans model is not initialized'}), 400
    if data is None:
        return jsonify({'error': 'Data is not generated'}), 400
    try:
        centroids, labels, converged = kmeans_model.step(data)
        # Convert NumPy boolean to Python boolean
        converged = bool(converged)
        return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist(), 'converged': converged})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run_to_convergence', methods=['POST'])
def run_to_convergence():
    global data, kmeans_model
    if kmeans_model is None:
        return jsonify({'error': 'KMeans model is not initialized'}), 400
    if data is None:
        return jsonify({'error': 'Data is not generated'}), 400
    try:
        converged = False
        while not converged:
            centroids, labels, converged = kmeans_model.step(data)
        # Convert NumPy boolean to Python boolean
        converged = bool(converged)
        return jsonify({'centroids': centroids.tolist(), 'labels': labels.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global data, kmeans_model, manual_centroids
    data = None
    kmeans_model = None
    manual_centroids = []
    return jsonify({'status': 'reset'})

@app.route('/get_state', methods=['GET'])
def get_state():
    global data, kmeans_model
    if data is None or kmeans_model is None:
        return jsonify({'error': 'KMeans not initialized'}), 400
    return jsonify({
        'data': data.tolist(),
        'centroids': kmeans_model.centroids.tolist(),
        'labels': kmeans_model.labels.tolist() if kmeans_model.labels is not None else None,
        'iteration': kmeans_model.iteration,
        'converged': bool(kmeans_model.converged)  # Convert NumPy boolean to Python boolean
    })

if __name__ == '__main__':
    app.run(debug=True)