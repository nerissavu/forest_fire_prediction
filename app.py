import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Wildfire Prediction System",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all saved models"""
    models = {}
    try:
        models['fire_size'] = pickle.load(open('fire_size_hectares_model.pkl', 'rb'))
        models['fire_duration'] = pickle.load(open('fire_duration_hours_model.pkl', 'rb'))
        models['suppression_cost'] = pickle.load(open('suppression_cost_$_model.pkl', 'rb'))
        models['fire_occurrence'] = pickle.load(open('fire_occurrence_model.pkl', 'rb'))
        models['feature_names'] = pickle.load(open('feature_names.pkl', 'rb'))
        return models, None
    except Exception as e:
        return None, str(e)

def create_input_dataframe(features):
    """Create a DataFrame from input features"""
    return pd.DataFrame([features])

def calculate_risk_level(fire_prob, fire_size, wind_speed, temperature):
    """Calculate overall risk level based on multiple factors"""
    risk_score = (
        0.4 * fire_prob + 
        0.3 * min(fire_size/100, 1) + 
        0.15 * (wind_speed/50) + 
        0.15 * (temperature/45)
    )
    
    if risk_score < 0.3:
        return "Low", "green"
    elif risk_score < 0.6:
        return "Medium", "orange"
    else:
        return "High", "red"

def plot_risk_factors(features, predictions):
    """Create a radar chart of risk factors"""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    # Normalize the factors to 0-1 scale
    factors = {
        'Temperature': features['Temperature (°C)'] / 50,
        'Wind Speed': features['Wind Speed (km/h)'] / 100,
        'Fire Prob': predictions['fire_prob'],
        'Humidity': 1 - (features['Humidity (%)'] / 100),  # Inverse as lower humidity = higher risk
        'Fire Size': min(predictions['fire_size'] / 100, 1)
    }
    
    # Prepare data for plotting
    angles = np.linspace(0, 2*np.pi, len(factors)+1)[:-1]
    values = list(factors.values())
    values.append(values[0])
    angles = np.concatenate((angles, [angles[0]]))
    
    # Plot
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(factors.keys())
    ax.set_ylim(0, 1)
    plt.title('Risk Factors Analysis')
    
    return fig

def main():
    # Load models
    models, error = load_models()
    if error:
        st.error(f"Error loading models: {error}")
        return

    # Title and description
    st.title("🔥 Wildfire Prediction System")
    st.markdown("""
    This system predicts various characteristics of potential wildfires based on environmental conditions.
    Enter the environmental parameters in the sidebar to get predictions.
    """)

    # Sidebar inputs
    st.sidebar.header("Environmental Parameters")
    
    # Weather inputs
    st.sidebar.subheader("Weather Conditions")
    temperature = st.sidebar.slider("Temperature (°C)", -10, 50, 25)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
    wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0, 100, 20)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 0.0, 0.1)
    
    # Terrain inputs
    st.sidebar.subheader("Terrain Characteristics")
    slope = st.sidebar.slider("Slope (%)", 0, 100, 15)
    
    # Location and vegetation inputs
    st.sidebar.subheader("Location and Vegetation")
    vegetation_type = st.sidebar.selectbox("Vegetation Type", 
        ["Forest", "Grassland", "Shrubland", "Mixed"])
    region = st.sidebar.selectbox("Region", 
        ["North", "South", "East", "West", "Central"])

    # Create input features dictionary
    features = {
        'Temperature (°C)': temperature,
        'Humidity (%)': humidity,
        'Wind Speed (km/h)': wind_speed,
        'Rainfall (mm)': rainfall,
        'Slope (%)': slope,
        'Vegetation Type': vegetation_type,
        'Region': region
    }

    # Create input DataFrame
    input_df = create_input_dataframe(features)

    # Make predictions
    try:
        predictions = {
            'fire_size': models['fire_size'].predict(input_df)[0],
            'fire_duration': models['fire_duration'].predict(input_df)[0],
            'suppression_cost': models['suppression_cost'].predict(input_df)[0],
            'fire_prob': models['fire_occurrence'].predict_proba(input_df)[0][1]
        }

        # Calculate risk level
        risk_level, risk_color = calculate_risk_level(
            predictions['fire_prob'], 
            predictions['fire_size'],
            wind_speed,
            temperature
        )

        # Display predictions in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fire Size", f"{predictions['fire_size']:.1f} hectares")
            st.metric("Fire Duration", f"{predictions['fire_duration']:.1f} hours")
        
        with col2:
            st.metric("Suppression Cost", f"${predictions['suppression_cost']:,.2f}")
            st.metric("Fire Probability", f"{predictions['fire_prob']:.1%}")
        
        with col3:
            st.markdown(f"""
            <div style='background-color: {risk_color}20; padding: 1rem; border-radius: 0.5rem; border: 1px solid {risk_color}'>
                <h3 style='color: {risk_color}'>Risk Level: {risk_level}</h3>
                <p>Based on current conditions</p>
            </div>
            """, unsafe_allow_html=True)

        # Plot risk analysis
        st.subheader("Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            risk_fig = plot_risk_factors(features, predictions)
            st.pyplot(risk_fig)
        
        with col2:
            st.markdown("### Key Risk Factors")
            risk_factors = [
                f"- Temperature: {'High' if temperature > 30 else 'Moderate' if temperature > 20 else 'Low'}",
                f"- Wind Speed: {'High' if wind_speed > 30 else 'Moderate' if wind_speed > 15 else 'Low'}",
                f"- Humidity: {'Low' if humidity < 30 else 'Moderate' if humidity < 60 else 'High'}",
                f"- Fire Probability: {'High' if predictions['fire_prob'] > 0.6 else 'Moderate' if predictions['fire_prob'] > 0.3 else 'Low'}"
            ]
            st.markdown("\n".join(risk_factors))

        # Recommendations
        if risk_level != "Low":
            st.subheader("Recommended Actions")
            recommendations = {
                "High": [
                    "🚨 Implement immediate fire prevention measures",
                    "👥 Alert local fire authorities",
                    "📋 Review and activate emergency response plans",
                    "🚫 Consider restricting access to high-risk areas",
                    "📡 Increase monitoring frequency"
                ],
                "Medium": [
                    "⚠️ Increase monitoring of the area",
                    "📝 Review fire prevention protocols",
                    "🔍 Inspect firefighting equipment",
                    "📱 Ensure communication systems are ready",
                    "🌳 Clear potential fuel sources"
                ]
            }
            for rec in recommendations[risk_level]:
                st.markdown(f"- {rec}")

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.write("Please check if all input parameters are within expected ranges.")

if __name__ == "__main__":
    main()