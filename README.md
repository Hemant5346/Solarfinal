```markdown
# ☀️ Solar Power Plant Data Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.12.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

An interactive web application for analyzing and visualizing solar power plant data. This tool provides insights into power output, classification of solar conditions, and custom data exploration capabilities.

## 🚀 Features

- Upload and process CSV files containing solar power plant data
- Visualize power output and solar conditions for specific dates
- Classify solar conditions using machine learning techniques
- Create custom graphs for any data column
- Download processed data and generated graphs

## 🛠 Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/solar-power-analysis.git
   cd solar-power-analysis
   ```

2. Create a Python 3.10 virtual environment:
   ```
   python3.10 -m venv env
   ```

3. Activate the virtual environment:
   - On macOS and Linux:
     ```
     source env/bin/activate
     ```
   - On Windows:
     ```
     .\env\Scripts\activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## 🖥 Usage

1. Ensure your virtual environment is activated.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to `http://localhost:8501`.

4. Upload your solar power plant CSV data and start analyzing!

## 📊 Data Format

The application expects CSV files with the following columns:
- `Time Stamp (local standard time) yyyy-mm-ddThh:mm:ss`
- `Pmp (W)`
- Other relevant solar power data columns

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/solar-power-analysis/issues).

## 📝 License




⭐️ Star this repo if you find it useful!
```