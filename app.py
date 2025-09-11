from flask import Flask, render_template, jsonify
import random
import csv
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# CSV file for storing patient history
CSV_FILE = "patient_history.csv"

# Create CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Pulse", "Temperature", "Oxygen", "Status"])

# Function to classify health status
def classify_health(pulse, temp, oxygen):
    if (60 <= pulse <= 100) and (36 <= temp <= 37.5) and (95 <= oxygen <= 100):
        return "Normal"
    elif (50 <= pulse <= 110) and (35 <= temp <= 38.5) and (90 <= oxygen <= 94):
        return "Alert"
    else:
        return "Critical"

# Home route -> Dashboard
@app.route("/")
def home():
    return render_template("index.html")

# API route -> Returns live data
@app.route("/get_data")
def get_data():
    pulse = random.randint(50, 110)
    temp = round(random.uniform(35, 39), 1)
    oxygen = random.randint(85, 100)

    status = classify_health(pulse, temp, oxygen)

    # Log data into CSV
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pulse, temp, oxygen, status])

    data = {
        "pulse": pulse,
        "temperature": temp,
        "oxygen": oxygen,
        "status": status
    }
    return jsonify(data)

# History route -> Shows table & chart
@app.route("/history")
def history():
    records = []
    with open(CSV_FILE, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            records.append(row)
    return render_template("history.html", records=records)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
