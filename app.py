from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

model = joblib.load("house_price_model.pkl")

# Define the CSV file path
CSV_FILE = "predictions.csv"

# Dummy credentials
USER_CREDENTIALS = {"kr624ayush@gmail.com": "123"}

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", user=session.get("user"))




@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email in USER_CREDENTIALS and USER_CREDENTIALS[email] == password:
            session["user"] = email  # Store session
            return redirect(url_for("home"))  # Redirect to home page

        return render_template("login.html", error="Invalid email or password!")

    return render_template("login.html")




@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))  # Redirect to login page




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

    except Exception as e:
        return render_template("index.html", prediction_text="Error: Invalid input!")




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
