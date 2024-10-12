import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import timedelta
import io
import datetime

# Set page configuration
st.set_page_config(page_title="Solar Power Analysis", layout="wide")

# Custom CSS to improve UI
st.markdown("""
<style>
    .stButton>button {
        color: #4F8BF9;
        border-radius: 50px;
        height: 3em;
        width: 100%;
    }
    .stDateInput>div>div>input {
        color: #4F8BF9;
    }
    .stSelectbox>div>div>select {
        color: #4F8BF9;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to load and preprocess data
def load_and_preprocess_data(file):
    df = pd.read_csv(file)
    df['Time Stamp'] = pd.to_datetime(df['Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss'])
    return df

# Function to analyze a single day
def analyze_day(day_data):
    nominal_power = day_data['Pmp (W)'].quantile(0.95)
    day_data['normalized_power'] = day_data['Pmp (W)'] / nominal_power
    day_data['hour'] = day_data['Time Stamp'].dt.hour
    day_data['minute'] = day_data['Time Stamp'].dt.minute
    day_data['time_of_day'] = day_data['hour'] + day_data['minute'] / 60
    
    peaks, _ = find_peaks(day_data['normalized_power'], height=0.5, distance=10)
    valleys, _ = find_peaks(-day_data['normalized_power'], height=-0.5, distance=10)
    
    avg_power = day_data['normalized_power'].mean()
    max_power = day_data['normalized_power'].max()
    power_variability = day_data['normalized_power'].std()
    peak_count = len(peaks)
    valley_count = len(valleys)
    
    smooth_threshold = 0.1
    power_changes = day_data['normalized_power'].diff().abs()
    smooth_periods = (power_changes <= smooth_threshold).astype(int)
    max_smooth_duration = smooth_periods.groupby(smooth_periods.ne(smooth_periods.shift()).cumsum()).sum().max() / len(day_data)
    
    morning_power = day_data[day_data['time_of_day'] < 12]['normalized_power'].mean()
    afternoon_power = day_data[day_data['time_of_day'] >= 12]['normalized_power'].mean()
    power_ratio = afternoon_power / morning_power if morning_power > 0 else 0
    
    peak_power_time = day_data.loc[day_data['normalized_power'].idxmax(), 'time_of_day']
    
    ramp_rates = day_data['normalized_power'].diff() / day_data['time_of_day'].diff()
    max_ramp_up = ramp_rates.max()
    max_ramp_down = ramp_rates.min()
    
    day_data['energy'] = day_data['Pmp (W)'] * (day_data['Time Stamp'].diff().dt.total_seconds() / 3600)
    total_energy = day_data['energy'].sum()
    
    return {
        'avg_power': avg_power,
        'max_power': max_power,
        'power_variability': power_variability,
        'peak_count': peak_count,
        'valley_count': valley_count,
        'max_smooth_duration': max_smooth_duration,
        'morning_power': morning_power,
        'afternoon_power': afternoon_power,
        'power_ratio': power_ratio,
        'peak_power_time': peak_power_time,
        'max_ramp_up': max_ramp_up,
        'max_ramp_down': max_ramp_down,
        'total_energy': total_energy,
        'day_data': day_data,
        'nominal_power': nominal_power
    }

# Function to classify solar conditions
def classify_solar_conditions(df):
    daily_data = df.groupby(df['Time Stamp'].dt.date)
    all_days_features = []

    for date, day_data in daily_data:
        features = analyze_day(day_data)
        all_days_features.append(features)

    X = pd.DataFrame([{k: v for k, v in f.items() if k != 'day_data' and k != 'nominal_power'} for f in all_days_features])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    cluster_map = {
        0: 'Mostly Sunny',
        1: 'Partly Sunny',
        2: 'Partly Cloudy',
        3: 'Mostly Cloudy'
    }

    y = pd.Series([cluster_map[c] for c in clusters])

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled, y)

    return clf, scaler, all_days_features, y


def format_time(x, _):
    hours, remainder = divmod(int(x), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}"

st.title('☀️ Solar Power Plant Data Analysis')

# Sidebar for file upload and date selection
with st.sidebar:
    st.header("📊 Data Input")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = load_and_preprocess_data(uploaded_file)
    clf, scaler, all_days_features, y = classify_solar_conditions(df)
    
    st.sidebar.success('🎉 File processed and classification completed!')
    
    # Display all available columns and let user select
    st.sidebar.header("🧮 Column Selection")
    all_columns = df.columns.tolist()
    selected_columns = st.sidebar.multiselect(
        "Select columns to display",
        options=all_columns,
        default=[]  # Changed to empty list so no columns are pre-selected
    )
    
    # Date selection in sidebar
    st.sidebar.header("📅 Date Selection")
    min_date = df['Time Stamp'].dt.date.min()
    max_date = df['Time Stamp'].dt.date.max()
    selected_date = st.sidebar.date_input('Select a date to visualize', 
                                          min_value=min_date, max_value=max_date, value=min_date)
    
    # Filter data for selected date and columns
    df_selected = df[df['Time Stamp'].dt.date == selected_date]
    
    if df_selected.empty:
        st.warning(f"No data available for the selected date: {selected_date}. Please choose another date.")
    else:
        df_display = df_selected[selected_columns] if selected_columns else pd.DataFrame()  # Empty DataFrame if no columns selected
        
        # Get features and classification for selected date
        selected_features = next(f for f in all_days_features if f['day_data']['Time Stamp'].dt.date.iloc[0] == selected_date)
        selected_classification = y.iloc[all_days_features.index(selected_features)]
        
        # Display summary statistics
        st.header("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Power", f"{selected_features['avg_power']:.2f}")
        col2.metric("Max Power", f"{selected_features['max_power']:.2f}")
        col3.metric("Classification", selected_classification)
        
        # Plot classification results
        st.header('🌤️ Solar Condition Classification')
        fig, ax = plt.subplots(figsize=(14, 7))

        time_in_seconds = df_selected['Time Stamp'].dt.hour * 3600 + df_selected['Time Stamp'].dt.minute * 60 + df_selected['Time Stamp'].dt.second

        ax.plot(time_in_seconds, df_selected['Pmp (W)'], label='Power Output')
        ax.set_xlabel('Time')
        ax.set_ylabel('Power Output (W)')
        ax.set_title(f'Solar Power Output on {selected_date}\nClassification: {selected_classification}')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
        plt.xticks(rotation=45)
        
        # Add nominal power line
        ax.axhline(y=selected_features['nominal_power'], color='r', linestyle='--', label='Nominal Power')
        
        # Highlight peaks and valleys
        peaks, _ = find_peaks(df_selected['Pmp (W)'], height=0.5*selected_features['nominal_power'], distance=10)
        valleys, _ = find_peaks(-df_selected['Pmp (W)'], height=-0.5*selected_features['nominal_power'], distance=10)
        ax.plot(time_in_seconds.iloc[peaks], df_selected['Pmp (W)'].iloc[peaks], "x", color='green', label='Peaks')
        ax.plot(time_in_seconds.iloc[valleys], df_selected['Pmp (W)'].iloc[valleys], "o", color='red', label='Valleys')
        
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)
        
        # Save classification plot as bytes object
        class_img_bytes = io.BytesIO()
        plt.savefig(class_img_bytes, format='png')
        class_img_bytes.seek(0)
        
        # Download button for the classification graph
        st.download_button(
            label="📥 Download Classification Graph as PNG",
            data=class_img_bytes,
            file_name=f"Solar_Classification_{selected_date}.png",
            mime="image/png"
        )
        
        # New graph with user-selected y-axis
        st.header('📈 Custom Graph')
        y_axis_options = [col for col in df_selected.columns if col != 'Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss']
        if y_axis_options:
            y_axis_column = st.selectbox("Select column for Y-axis", options=y_axis_options)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.plot(df_selected['Time Stamp'], df_selected[y_axis_column])
            ax.set_xlabel('Time')
            ax.set_ylabel(y_axis_column)
            ax.set_title(f'{y_axis_column} vs Time on {selected_date}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
          
            custom_img_bytes = io.BytesIO()
            plt.savefig(custom_img_bytes, format='png')
            custom_img_bytes.seek(0)
            
            # Download button for the custom graph
            st.download_button(
                label="📥 Download Custom Graph as PNG",
                data=custom_img_bytes,
                file_name=f"{y_axis_column}_vs_Time_{selected_date}.png",
                mime="image/png"
            )
        else:
            st.info("No numeric columns available for plotting. Please select columns in the sidebar.")
        
        # Display classified data for the selected date with selected columns
        st.header(f'📅 Classified Data for {selected_date}')
        if not df_display.empty:
            st.dataframe(df_display)
        else:
            st.info("Please select columns to display data.")
        
        # Option to download processed data for the selected date with selected columns
        if not df_display.empty:
            csv_selected = df_display.to_csv(index=False)
            st.download_button(
                label=f"📥 Download processed data for {selected_date} as CSV",
                data=csv_selected,
                file_name=f"processed_solar_data_{selected_date}.csv",
                mime="text/csv",
            )

else:
    st.info('👆 Please upload a CSV file in the sidebar to begin analysis.')

# Footer
st.markdown("---")
