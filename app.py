from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import os
from authlib.integrations.flask_client import OAuth

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key")  # Use environment variable for security

# Load the model
MODEL_PATH = os.path.join(os.getcwd(), "house_price_model.pkl")
model = joblib.load(MODEL_PATH)

# Define the CSV file path
CSV_FILE = "predictions.csv"

# Google OAuth Configuration (Replace with actual credentials)
app.config["GOOGLE_CLIENT_ID"] = os.environ.get("GOOGLE_CLIENT_ID")  
app.config["GOOGLE_CLIENT_SECRET"] = os.environ.get("GOOGLE_CLIENT_SECRET")

oauth = OAuth(app)
google = oauth.register(
    name="google",
    client_id=app.config["GOOGLE_CLIENT_ID"],
    client_secret=app.config["GOOGLE_CLIENT_SECRET"],
    access_token_url="https://oauth2.googleapis.com/token",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    client_kwargs={"scope": "openid email profile"},
)

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session.get("user"))

@app.route("/login")
def google_login():
    return google.authorize_redirect(url_for("authorize", _external=True, _scheme="https"))  # Ensure HTTPS in production

@app.route("/authorize")
def authorize():
    token = google.authorize_access_token()
    user_info = google.parse_id_token(token)
    session["user"] = user_info["email"]
    return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        floors = int(request.form["floors"])
        condition = int(request.form["condition"])

        # Prepare input for model prediction
        input_features = np.array([[area, bedrooms, bathrooms, floors, condition]])
        prediction = model.predict(input_features)[0]

        # Save the data to CSV
        new_data = pd.DataFrame([[area, bedrooms, bathrooms, floors, condition, prediction]],
                                columns=["Area", "Bedrooms", "Bathrooms", "Floors", "Condition", "Predicted Price"])

        if not os.path.isfile(CSV_FILE):
            new_data.to_csv(CSV_FILE, index=False)  # Create file with header if it doesn't exist
        else:
            new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)  # Append to existing file

        return render_template("index.html", prediction_text=f"Estimated House Price: Rs {prediction:,.2f}")

    except Exception:
        return render_template("index.html", prediction_text="Error: Invalid input!")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use dynamic port for deployment
    app.run(host="0.0.0.0", port=port)
