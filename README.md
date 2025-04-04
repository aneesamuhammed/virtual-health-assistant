# 🩺 Virtual Health Assistant

A web-based AI assistant for predicting diseases based on user symptoms and showing real-time simulated vitals like blood pressure and heart rate. Built using Python, Flask, TensorFlow, and JavaScript.

---

## 📁 Project Structure

virtual-health-assistant/ │ ├── init.py # (Optional) Package initializer if this is part of a module ├── train.py # Trains the deep learning model on symptoms/diseases dataset ├── app.py # Main Flask application (backend API and HTML rendering) ├── vitals_generator.py # Simulates patient vitals (e.g., BP, Heart Rate) ├── templates/ │ └── index.html # Frontend web UI (form, chart, vitals display) ├── static/ │ └── icons/ # Icons for vitals (heart, blood pressure, etc.) ├── model/ │ └── chatbot_model.hdf5 # Trained model file ├── data/ │ ├── words.pkl # Vocabulary used for input preprocessing │ └── labels.pkl # Disease label mappings └── README.md # You're here!

---

## 🚀 How It Works

1. User enters symptoms into the web UI.
2. Symptoms are sent to the Flask backend (`/predict` route).
3. The trained model (`chatbot_model.hdf5`) predicts the top 5 probable diseases.
4. A pie chart shows the disease probabilities visually.
5. Simulated vitals are displayed alongside icons (heart rate, BP, etc.).

---


python init.py
Train the model (optional if model already present):
python train.py
Run the Flask app:
python app.py

Open browser and visit: http://127.0.0.1:5000

📄 File Descriptions

File	Description
init.py	Optional file if packaging; initializes Python module
train.py	Loads and processes training data, builds, trains, and saves the TensorFlow model
app.py	Flask backend that handles routes, predictions, and returns JSON/HTML
vitals_generator.py	Randomly generates simulated patient vitals (heart rate, BP, oxygen)
index.html	The user interface with form, pie chart, and vitals display
chatbot_model.hdf5	Saved TensorFlow model for symptom prediction
words.pkl, labels.pkl	Serialized data from preprocessing used for predictions


Features

🎯 Predicts top 5 most likely diseases
📈 Pie chart visualization of probabilities
❤️ Simulated vitals with icons
⚡ Real-time interaction (AJAX, no page reloads)
🧠 Future Enhancements

Integration with real-world vitals sensors
Improved accuracy based on training 
Home remedies suggestions
Detailed symptoms and other facts of predicted disease
Doctor consultation booking


🛡️ Disclaimer

This project is for educational purposes only and not intended to provide medical advice. Always consult a healthcare professional for real diagnosis and treatment.





