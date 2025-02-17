from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

def hex_to_rgb(hex_color):
    """Convert hex color code to RGB tuple."""
    hex_color = hex_color.lstrip("#")  # Remove '#' if present
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Load dataset
file_path = "Final_Topwear_Bottomwear_Color_Combination.csv"
df = pd.read_csv(file_path)

# Encode categorical features numerically
trend_mapping = {"Low": 1, "Medium": 2, "High": 3}
df["Trend_Alignment"] = df["Trend_Alignment"].map(trend_mapping)

# Convert bottomwear colors to RGB values
df["Bottomwear_RGB"] = df["Bottomwear_Hex"].apply(hex_to_rgb)

# Prepare data for KNN
X = np.array(df["Bottomwear_RGB"].tolist())  # RGB values of bottomwear

# Train a K-Nearest Neighbors (KNN) model
knn = NearestNeighbors(n_neighbors=3, metric="euclidean")  # 3 nearest matches
knn.fit(X)  # Train model on bottomwear RGB colors

@app.route("/recommend", methods=["POST"])
def recommend():
    """API Endpoint to get topwear recommendations based on bottomwear color."""
    data = request.json  # Get input color from request
    bottomwear_hex = data["color"]  # Extract hex code
    bottomwear_rgb = np.array(hex_to_rgb(bottomwear_hex)).reshape(1, -1)  # Convert to RGB

    distances, indices = knn.kneighbors(bottomwear_rgb)  # Find nearest neighbors
    recommendations = df.iloc[indices[0]][["Topwear_Color_Name", "Season", "Occasion", "Trend_Alignment", "Mood_Conveyed","Time_of_Day_Preference","Skin_Tone_Compatibility"]]

    return jsonify(recommendations.to_dict(orient="records"))  # Convert DataFrame to JSON

if __name__ == "__main__":
    # Run the Flask app on port 8000
    app.run(debug=True, host="0.0.0.0", port=8000)
