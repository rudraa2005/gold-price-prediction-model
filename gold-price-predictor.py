import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Load data
db = pd.read_csv('Gold_price.csv', dtype={
    'priceOpen': str,
    'priceHigh': str,
    'priceLow': str,
    'priceClose': str
})

# Convert comma to dot
for col in ['priceOpen', 'priceHigh', 'priceLow', 'priceClose']:
    db[col] = db[col].str.replace(',', '.').astype(float)

# Convert timestamps to datetime
db['Date'] = pd.to_datetime(db['timeOpen'], unit='ms')
db.sort_values('Date', inplace=True)
db.set_index('Date', inplace=True)

# Enhanced Feature Engineering
def create_technical_indicators(df):
    """Create technical indicators for better prediction"""
    # Price-based features
    df['price_range'] = df['priceHigh'] - df['priceLow']
    df['price_change'] = df['priceClose'] - df['priceOpen']
    df['price_change_pct'] = (df['priceClose'] - df['priceOpen']) / df['priceOpen'] * 100
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'ma_{window}'] = df['priceClose'].rolling(window=window).mean()
        df[f'price_vs_ma_{window}'] = df['priceClose'] / df[f'ma_{window}'] - 1
    
    # Volatility indicators
    df['volatility_5'] = df['priceClose'].rolling(window=5).std()
    df['volatility_20'] = df['priceClose'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = df['priceClose'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['priceClose'].rolling(window=20).mean()
    bb_std = df['priceClose'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['priceClose'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # MACD
    ema_12 = df['priceClose'].ewm(span=12).mean()
    ema_26 = df['priceClose'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Lag features (multiple time steps)
    for lag in [1, 2, 3, 5, 10]:
        df[f'close_lag_{lag}'] = df['priceClose'].shift(lag)
        df[f'volume_lag_{lag}'] = df.get('volume', pd.Series(index=df.index, data=1)).shift(lag)
        df[f'change_lag_{lag}'] = df['price_change'].shift(lag)
    
    # Time-based features
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    
    # Cyclical encoding of time features
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

# Apply feature engineering
db = create_technical_indicators(db)

# Create target variable
db['target'] = db['priceClose'].shift(-1)

# Drop rows with NaN values
db.dropna(inplace=True)

# Define comprehensive feature set
feature_columns = [
    # Basic OHLC
    'priceOpen', 'priceHigh', 'priceLow', 'priceClose',
    # Price features
    'price_range', 'price_change', 'price_change_pct',
    # Moving averages and ratios
    'ma_5', 'ma_10', 'ma_20', 'ma_50',
    'price_vs_ma_5', 'price_vs_ma_10', 'price_vs_ma_20', 'price_vs_ma_50',
    # Technical indicators
    'volatility_5', 'volatility_20', 'rsi', 'bb_position', 'macd', 'macd_signal',
    # Lag features
    'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5', 'close_lag_10',
    'change_lag_1', 'change_lag_2', 'change_lag_3', 'change_lag_5', 'change_lag_10',
    # Time features
    'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'quarter'
]

# Filter features that exist in the dataframe
features = [col for col in feature_columns if col in db.columns]

# Split data (use more recent data for testing)
train_size = int(0.8 * len(db))
train_df = db.iloc[:train_size]
test_df = db.iloc[train_size:]

X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and ensemble them
models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=200, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    ),
    'LinearRegression': LinearRegression()
}

# Train models and make predictions
model_predictions = {}
model_scores = {}

print("--- Model Training and Evaluation ---")
for name, model in models.items():
    if name == 'LinearRegression':
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    
    model_predictions[name] = pred
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    model_scores[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.8f}")
    print(f"  MAE: {mae:.8f}")
    print(f"  R² Score: {r2:.4f}")

# Ensemble prediction (weighted average based on R² scores)
weights = np.array([max(0, model_scores[name]['R2']) for name in models.keys()])
weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)

ensemble_pred = np.average([model_predictions[name] for name in models.keys()], 
                          axis=0, weights=weights)

# Evaluate ensemble
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

print(f"\n--- Ensemble Results ---")
print(f"RMSE: {ensemble_rmse:.8f}")
print(f"MAE: {ensemble_mae:.8f}")
print(f"R² Score: {ensemble_r2:.4f}")

# Future prediction with uncertainty estimation
def predict_future_with_uncertainty(models, scaler, last_data, n_days=180, n_simulations=20):
    """Predict future prices with uncertainty estimation (optimized)"""
    future_predictions = []
    prediction_intervals = []
    
    last_date = db.index[-1]
    
    # Pre-calculate some values for efficiency
    recent_volatility = db['volatility_5'].iloc[-10:].mean()
    weights_array = np.array(weights)
    
    print("Progress: ", end="")
    for day in range(n_days):
        if day % 30 == 0:  # Progress indicator
            print(f"{day//30*30}d", end=" ", flush=True)
            
        future_date = last_date + timedelta(days=day+1)
        daily_predictions = []
        
        # Vectorized simulation generation for speed
        price_noise = np.random.normal(0, recent_volatility * 0.5, n_simulations)
        range_multipliers = np.random.uniform(0.8, 1.2, n_simulations)
        high_factors = np.random.uniform(0.3, 0.7, n_simulations)
        low_factors = np.random.uniform(0.3, 0.7, n_simulations)
        
        # Time features (constant for all simulations)
        dayofweek_sin = np.sin(2 * np.pi * future_date.dayofweek / 7)
        dayofweek_cos = np.cos(2 * np.pi * future_date.dayofweek / 7)
        month_sin = np.sin(2 * np.pi * future_date.month / 12)
        month_cos = np.cos(2 * np.pi * future_date.month / 12)
        quarter = future_date.quarter
        
        for sim in range(n_simulations):
            # Create synthetic OHLC data
            next_open = last_data['priceClose'] * (1 + price_noise[sim])
            daily_range = last_data['price_range'] * range_multipliers[sim]
            next_high = next_open + daily_range * high_factors[sim]
            next_low = next_open - daily_range * low_factors[sim]
            
            # Create input data efficiently
            input_data = last_data.copy()
            input_data['priceOpen'] = next_open
            input_data['priceHigh'] = next_high
            input_data['priceLow'] = next_low
            input_data['dayofweek_sin'] = dayofweek_sin
            input_data['dayofweek_cos'] = dayofweek_cos
            input_data['month_sin'] = month_sin
            input_data['month_cos'] = month_cos
            input_data['quarter'] = quarter
            
            # Make predictions with each model
            sim_predictions = []
            feature_values = input_data[features].values.reshape(1, -1)
            
            # Random Forest prediction
            rf_pred = models['RandomForest'].predict(feature_values)[0]
            sim_predictions.append(rf_pred)
            
            # Gradient Boosting prediction
            gb_pred = models['GradientBoosting'].predict(feature_values)[0]
            sim_predictions.append(gb_pred)
            
            # Linear Regression prediction
            lr_pred = models['LinearRegression'].predict(scaler.transform(feature_values))[0]
            sim_predictions.append(lr_pred)
            
            # Ensemble prediction for this simulation
            ensemble_sim_pred = np.average(sim_predictions, weights=weights_array)
            daily_predictions.append(ensemble_sim_pred)
        
        # Calculate statistics for this day
        daily_predictions = np.array(daily_predictions)
        mean_pred = np.mean(daily_predictions)
        lower_bound = np.percentile(daily_predictions, 5)  # 90% confidence interval
        upper_bound = np.percentile(daily_predictions, 95)
        
        future_predictions.append(mean_pred)
        prediction_intervals.append((lower_bound, upper_bound))
        
        # Update last_data for next iteration
        last_data['priceClose'] = mean_pred
        # Update lag features efficiently
        if 'close_lag_10' in last_data:
            for lag in range(10, 1, -1):
                if f'close_lag_{lag}' in last_data and f'close_lag_{lag-1}' in last_data:
                    last_data[f'close_lag_{lag}'] = last_data[f'close_lag_{lag-1}']
        if 'close_lag_1' in last_data:
            last_data['close_lag_1'] = last_data['priceClose']
    
    print("Complete!")
    return future_predictions, prediction_intervals

# Generate future predictions (optimized)
print("\n--- Generating Future Predictions (180 days) ---")
print("This may take 30-60 seconds... Please wait.")
last_known = db[features].iloc[-1].copy()
future_dates = pd.date_range(start=db.index[-1] + pd.Timedelta(days=1), periods=180)

# Reduce simulations for faster execution (you can increase back to 50-100 for production)
future_preds, intervals = predict_future_with_uncertainty(
    models, scaler, last_known, n_days=180, n_simulations=20
)

# Print sample of future predictions
print("\n--- Sample Future Predicted Prices (Next 30 Days) ---")
for i in range(min(30, len(future_dates))):
    date = future_dates[i]
    price = future_preds[i]
    lower, upper = intervals[i]
    print(f"{date.date()} -> {price:.4f} (95% CI: {lower:.4f} - {upper:.4f})")

# Enhanced plotting
plt.figure(figsize=(15, 10))

# Plot 1: Model comparison on test data
plt.subplot(2, 2, 1)
plt.plot(y_test.index, y_test.values, label='Actual', linewidth=2, alpha=0.8)
for name, pred in model_predictions.items():
    plt.plot(y_test.index, pred, label=f'{name} (R²={model_scores[name]["R2"]:.3f})', alpha=0.7)
plt.plot(y_test.index, ensemble_pred, label=f'Ensemble (R²={ensemble_r2:.3f})', 
         linewidth=2, linestyle='--')
plt.title("Model Comparison on Test Data")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Residuals analysis
plt.subplot(2, 2, 2)
residuals = y_test.values - ensemble_pred
plt.scatter(ensemble_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals Plot (Ensemble)")
plt.grid(True, alpha=0.3)

# Plot 3: Future predictions with uncertainty
plt.subplot(2, 1, 2)
# Plot historical data (last 100 days)
recent_data = db.iloc[-100:]
plt.plot(recent_data.index, recent_data['priceClose'], 
         label='Historical', color='blue', linewidth=2)

# Plot future predictions
plt.plot(future_dates, future_preds, 
         label='Future Predictions', color='red', linewidth=2)

# Plot confidence intervals
lower_bounds = [interval[0] for interval in intervals]
upper_bounds = [interval[1] for interval in intervals]
plt.fill_between(future_dates, lower_bounds, upper_bounds, 
                alpha=0.3, color='red', label='90% Confidence Interval')

plt.title("Gold Price Prediction with Uncertainty (Next 6 Months)")
plt.xlabel("Date")
plt.ylabel("Gold Price")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Feature importance analysis
if 'RandomForest' in models:
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': models['RandomForest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n--- Top 10 Most Important Features ---")
    print(feature_importance.head(10).to_string(index=False))

print(f"\n--- Summary Statistics ---")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {len(features)}")
print(f"Best model: {max(model_scores.keys(), key=lambda x: model_scores[x]['R2'])}")
print(f"Ensemble improvement over best single model: {ensemble_r2 - max(model_scores[x]['R2'] for x in model_scores):.4f}")
