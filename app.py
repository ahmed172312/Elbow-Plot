st.pyplot(fig)
st.write('Pick the number of clusters where the inertia starts to decrease more slowly (the "elbow").')
from flask import Flask, request, send_file, send_from_directory
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import os
app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory(os.path.abspath(os.path.dirname(__file__)), 'index.html')

@app.route('/elbow')
def elbow():
    max_clusters = int(request.args.get('max_clusters', 10))
    X, _ = make_blobs(n_samples=500, centers=5, random_state=42)
    inertia = []
    K = range(1, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K, inertia, 'bo-')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Plot')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)
