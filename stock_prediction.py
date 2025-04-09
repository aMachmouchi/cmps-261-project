# %% [markdown]
# # S&P 500 Stock Price Prediction
# 
# This script implements a machine learning project to predict S&P 500 stock prices. The analysis includes:
# - Exploratory Data Analysis (EDA)
# - Data cleaning and preprocessing
# - Feature selection
# - Model implementation and comparison (Linear Regression and LSTM)

# %%
# Import Required Libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import os
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Data Collection and EDA

# %%
def download_data():
    """Download S&P 500 data or load from CSV if available"""
    if os.path.exists("sp500.csv") and os.path.getsize("sp500.csv") > 0:
        sp500 = pd.read_csv("sp500.csv", index_col=0)
        print(f"Data loaded successfully from CSV. Shape: {sp500.shape}")
    else:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        sp500.to_csv("sp500.csv")
        print(f"Data fetched successfully from yfinance. Shape: {sp500.shape}")
    
    sp500.index = pd.to_datetime(sp500.index, utc=True)
    return sp500

def add_technical_indicators(df):
    """Add technical indicators to the dataset"""
    # Calculate returns
    df['Returns'] = df['Close'].pct_change()
    
    # Calculate moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate exponential moving averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Calculate MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Calculate Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Calculate Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Calculate price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Calculate price volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calculate price range
    df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
    
    # Calculate price change
    df['Price_Change'] = df['Close'].diff()
    
    # Drop rows with NaN values (from rolling calculations)
    df = df.dropna()
    
    return df

# Download data
df = download_data()

# Add technical indicators
print("\nAdding technical indicators...")
df = add_technical_indicators(df)

# Data Cleaning
print("\nInitial data shape:", df.shape)

# Calculate percentage of missing values in each column
missing_cols = df.isnull().mean() * 100
print("\nPercentage of missing values in each column:")
print(missing_cols)

# Remove columns with more than 70% missing values
cols_to_drop = missing_cols[missing_cols > 70].index
if len(cols_to_drop) > 0:
    print(f"\nRemoving columns with >70% missing values: {list(cols_to_drop)}")
    df = df.drop(columns=cols_to_drop)

# Calculate percentage of missing values in each row
missing_rows = df.isnull().mean(axis=1) * 100

# Remove rows with more than 70% missing values
rows_before = len(df)
df = df[missing_rows <= 70]
rows_removed = rows_before - len(df)
print(f"\nRemoved {rows_removed} rows with >70% missing values")

# Fill remaining missing values using interpolation
print("\nFilling remaining missing values using interpolation...")
df_before_interpolation = df.copy()
df = df.interpolate(method='linear', limit_direction='both')

# Calculate how many values were interpolated
total_interpolated = (df_before_interpolation.isnull().sum().sum())
print(f"Interpolated {total_interpolated} missing values")

print("\nFinal data shape:", df.shape)

# Check if any missing values remain
remaining_missing = df.isnull().sum().sum()
if remaining_missing > 0:
    print(f"\nWarning: {remaining_missing} missing values remain after cleaning")
else:
    print("\nAll missing values have been handled")

# Perform EDA
print("\nBasic Statistics after cleaning:")
print(df.describe())

# Plot price history
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Close'])
plt.title('S&P 500 Historical Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Plot volume
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Volume'])
plt.title('S&P 500 Trading Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.grid(True)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.show()

# Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Close'], kde=True)
plt.title('Distribution of Closing Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Daily returns
df['Returns'] = df['Close'].pct_change()
plt.figure(figsize=(10, 6))
sns.histplot(df['Returns'].dropna(), kde=True)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

print("\nFirst few rows of the data:")
print(df.head())

# %% [markdown]
# ## 2. Data Cleaning and Feature Selection

# %%
def prepare_data(df):
    """Clean data and select features"""
    # Clean data
    df = df.dropna()
    
    # Select features - now including technical indicators
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_5', 'SMA_20', 'SMA_50',
                'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal', 'RSI', 'BB_Middle',
                'BB_Upper', 'BB_Lower', 'ATR', 'Volume_SMA', 'Volume_Change',
                'Momentum', 'Volatility', 'Price_Range', 'Price_Change']
    target = 'Close'
    
    print("\nSelected features:", features)
    print("Target variable:", target)
    
    return df[features], df[target]

# Prepare data
X, y = prepare_data(df)

# Feature Importance Analysis using Random Forest
print("\nAnalyzing feature importance using Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance scores
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Scores:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Select features with importance > 0.01 (1%)
important_features = feature_importance[feature_importance['Importance'] > 0.01]['Feature'].tolist()
print(f"\nSelected important features: {important_features}")

# Update X with only important features
X = X[important_features]

def prepare_lstm_data(X, y, time_steps=30):
    """Prepare data for LSTM model"""
    X_lstm, y_lstm = [], []
    for i in range(len(X) - time_steps):
        X_lstm.append(X[i:(i + time_steps)])
        y_lstm.append(y[i + time_steps])
    return np.array(X_lstm), np.array(y_lstm)

def create_lstm_model(input_shape):
    """Create simplified LSTM model"""
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Scale features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# Split data
train_size = int(len(X_scaled) * 0.8)
X_train = X_scaled[:train_size]
X_test = X_scaled[train_size:]
y_train = y_scaled[:train_size]
y_test = y_scaled[train_size:]

# Prepare LSTM data
time_steps = 30
X_train_lstm, y_train_lstm = prepare_lstm_data(X_train, y_train, time_steps)
X_test_lstm, y_test_lstm = prepare_lstm_data(X_test, y_test, time_steps)

# Train Linear Regression
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
y_pred_lr = scaler_y.inverse_transform(y_pred_lr.reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test)

# Calculate metrics for Linear Regression
mse_lr = mean_squared_error(y_test_actual, y_pred_lr)
r2_lr = r2_score(y_test_actual, y_pred_lr)

# Train LSTM
print("\nTraining LSTM...")
lstm_model = create_lstm_model((time_steps, X_train_lstm.shape[2]))
history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, verbose=1)

# Make predictions with LSTM
y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm)
y_test_lstm_actual = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1))

# Calculate metrics for LSTM
mse_lstm = mean_squared_error(y_test_lstm_actual, y_pred_lstm)
r2_lstm = r2_score(y_test_lstm_actual, y_pred_lstm)

# Store results
results = {
    'Linear Regression': {'MSE': mse_lr, 'R2': r2_lr},
    'LSTM': {'MSE': mse_lstm, 'R2': r2_lstm}
}

# Print results
print("\nModel Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"R2 Score: {metrics['R2']:.4f}")
    
# Create a comparison bar plot
plt.figure(figsize=(10, 6))
models = list(results.keys())
mse_values = [results[model]['MSE'] for model in models]
r2_values = [results[model]['R2'] for model in models]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

rects1 = ax1.bar(x - width/2, mse_values, width, label='MSE')
rects2 = ax2.bar(x + width/2, r2_values, width, label='R2 Score', color='orange')

ax1.set_xlabel('Models')
ax1.set_ylabel('MSE', color='blue')
ax2.set_ylabel('R2 Score', color='orange')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show() 