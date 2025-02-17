from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

def hex_to_rgb(hex_color):
    """Convert hex color code to RGB tuple safely."""
    try:
        hex_color = hex_color.lstrip("#")  # Remove '#' if present
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return None  # Invalid hex code

# Load dataset
file_path = "Final_Topwear_Bottomwear_Color_Combination.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found. Make sure it's in the correct location.")
    exit()

# Encode categorical features numerically
trend_mapping = {"Low": 1, "Medium": 2, "High": 3}
df["Trend_Alignment"] = df["Trend_Alignment"].map(trend_mapping)

# Convert bottomwear colors to RGB values
df["Bottomwear_RGB"] = df["Bottomwear_Hex"].apply(lambda x: hex_to_rgb(x) if isinstance(x, str) else None)
df = df.dropna(subset=["Bottomwear_RGB"])  # Drop rows with invalid RGB conversion

# Prepare data for KNN
X = np.array(df["Bottomwear_RGB"].tolist())  # RGB values of bottomwear

# Train a K-Nearest Neighbors (KNN) model
knn = NearestNeighbors(n_neighbors=min(3, len(X)), metric="euclidean")  # Avoid errors if dataset has fewer items
knn.fit(X)  # Train model on bottomwear RGB colors

@app.route("/recommend", methods=["POST"])
def recommend():
    """API Endpoint to get topwear recommendations based on bottomwear color."""
    data = request.json

    # Validate input
    if not data or "color" not in data:
        return jsonify({"error": "Missing 'color' in request body"}), 400

    bottomwear_hex = data["color"]
    bottomwear_rgb = hex_to_rgb(bottomwear_hex)

    if bottomwear_rgb is None:
        return jsonify({"error": "Invalid hex color format"}), 400

    bottomwear_rgb = np.array(bottomwear_rgb).reshape(1, -1)  # Convert to RGB array

    # Get nearest neighbors
    distances, indices = knn.kneighbors(bottomwear_rgb)
    
    # Fetch recommendations with Topwear Color Hex and Color Name
    recommendations = df.iloc[indices[0]][[
        "Topwear_Color_Name", "Topwear_Hex", "Season", "Occasion",
        "Trend_Alignment", "Mood_Conveyed", "Time_of_Day_Preference",
        "Skin_Tone_Compatibility"
    ]]

    return jsonify(recommendations.to_dict(orient="records"))  # Convert DataFrame to JSON

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
