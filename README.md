# 🔥 Wildfire Prediction System

A machine learning-based system that predicts various characteristics of potential wildfires based on environmental conditions. The system uses multiple models to predict fire size, duration, suppression cost, and occurrence probability.

## 📋 Features

- Real-time prediction of:
  - Fire size (hectares)
  - Fire duration (hours)
  - Suppression cost ($)
  - Fire occurrence probability
- Interactive visualization of risk factors
- Risk level assessment
- Customized recommendations based on risk level



## 🚀 Usage
1. Access Website here: https://forestfireprediction-ngavu.streamlit.app/

2. Input environmental parameters:
   - Temperature
   - Humidity
   - Wind Speed
   - Rainfall
   - Slope
   - Vegetation Type
   - Region

3. View predictions and recommendations in real-time

## 📚 Project Structure

```
wildfire-prediction/
├── app.py                 # Streamlit application
├── requirements.txt       # Project dependencies
├── README.md             # Project documentation
└── saved_models/         # Trained model files
    ├── fire_size_model.pkl
    ├── fire_duration_model.pkl
    ├── suppression_cost_model.pkl
    └── fire_occurrence_model.pkl
```

## 🔄 Model Training

The system uses several machine learning models:
- Random Forest
- XGBoost
- Decision Tree
- Support Vector Machine (for classification)

Models are trained on historical wildfire data considering various environmental factors.

## 📊 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Temperature | Numerical | Temperature in Celsius |
| Humidity | Numerical | Relative humidity percentage |
| Wind Speed | Numerical | Wind speed in km/h |
| Rainfall | Numerical | Rainfall in mm |
| Slope | Numerical | Terrain slope percentage |
| Vegetation Type | Categorical | Type of vegetation in the area |
| Region | Categorical | Geographical region |

## 📈 Output Predictions

- **Fire Size**: Predicted area affected by the fire in hectares
- **Fire Duration**: Expected duration of the fire in hours
- **Suppression Cost**: Estimated cost of fire suppression in dollars
- **Fire Occurrence**: Probability of fire occurrence
- **Risk Level**: Overall risk assessment (Low, Medium, High)

## ✅ Deon Checklist
https://huggingface.co/spaces/ngavu2592/Oct18Ethic 

## 💡 Technologies Used

- Python 3.10+
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn




