# CMPS 261 Project: Stock Price Prediction

This project implements a machine learning system for predicting S&P 500 stock prices using both traditional machine learning and deep learning approaches.

## Features

- Data collection and preprocessing
- Feature engineering with technical indicators
- Feature importance analysis using Random Forest
- Multiple model implementations:
  - Linear Regression
  - LSTM (Long Short-Term Memory) neural network
- Performance evaluation and visualization

## Technical Indicators

The project includes the following technical indicators:
- Moving Averages (SMA, EMA)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- Volume indicators
- Price momentum and volatility measures

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - yfinance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cmps-261-project.git
cd cmps-261-project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python stock_prediction.py
```

## Results

The model achieves the following performance metrics:
- Linear Regression: MSE = 211.46, R² = 0.9999
- LSTM: MSE = 3947.16, R² = 0.9977

## License

This project is licensed under the MIT License - see the LICENSE file for details. 