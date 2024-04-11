import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from cryptography.fernet import Fernet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
import ccxt
import matplotlib.animation as animation
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import numpy as np
import re

# Define the crypto symbols
ticker_symbols = ['BTC-USD', 'ETH-USD']

# Generate an encryption key
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Function to sanitise input data
def sanitize_input(input_data):
    # Use regular expressions to remove unwanted characters or patterns
    sanitized_data = re.sub(r'[^\w\s]', '', input_data)
    return sanitized_data

# Function to validate ticker symbols
def validate_ticker_symbols(symbols):
    # Define a pattern for valid ticker symbols (e.g., 'BTC-USD', 'ETH-USD')
    pattern = re.compile(r'^[A-Z]{3,5}-USD$')
    for symbol in symbols:
        if not pattern.match(symbol):
            raise ValueError(f"Invalid ticker symbol: {symbol}")
    return symbols

# Define the crypto symbols and validate them
ticker_symbols = validate_ticker_symbols(['BTC-USD', 'ETH-USD'])

# Function to preprocess the data
def preprocess_data(data):
    # Step 1: Data Cleaning
    # Remove any rows with missing values (NaN, null) from the dataset.
    data.dropna(inplace=True)

    # Step 2: Data Scaling
    # Use Min-Max scaling to transform the numerical features to a common scale.
    # This is done to ensure that each feature contributes equally to the analysis.
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

    # Step 3: Feature Engineering
    # Calculate the 50-day Moving Average (MA) of the 'Close' price.
    data['50_MA'] = data['Close'].rolling(window=50).mean()

    # Step 4: Normalisation
    # Normalise the 'Volume' feature using Min-Max scaling.
    # This will transform the 'Volume' feature to a common scale similar to the other features.
    data['Volume'] = scaler.fit_transform(data[['Volume']])

    # Step 5: Adding Polynomial Features
    # Use PolynomialFeatures to create new polynomial features from the original features.
    # This can capture non-linear relationships between the features and the target variable.
    poly = PolynomialFeatures(degree=2)
    data_poly = poly.fit_transform(data_scaled)

    # Step 6: Handling Missing Values
    # Impute any remaining missing values with the mean of the respective column.
    # This ensures that the dataset is complete and ready for further analysis.
    data = data.fillna(data.mean())

    # Step 7: Feature if you had potential domain knowledge
    # Add new features based on domain knowledge, such as 'Price Change' (Close - Open).
    # These features can provide valuable insights for the machine learning models.
    data['Price Change'] = data['Close'] - data['Open']

    return data, data_poly


# Get the date 2 years ago from today
start_date = datetime.now() - timedelta(days=2 * 365)

# Run the data retrieval process daily
while datetime.now() < (start_date + timedelta(days=2 * 365) + timedelta(days=1)):
    for symbol in ticker_symbols:

        # Sanitise the symbol before using it in file names or any other operations
        sanitized_symbol = sanitize_input(symbol)

        # Get the data for the crypto symbols
        data = yf.download(symbol, start=start_date, end=datetime.now().date())

        # Save the raw, unprocessed data to a CSV file
        data.to_csv(f'{symbol}_raw.csv')

        # Preprocess the data
        preprocessed_data, preprocessed_data_poly = preprocess_data(data)

        # Save the preprocessed data to a CSV file
        preprocessed_data.to_csv(f'{symbol}_preprocessed.csv')

        # Save the preprocessed data with polynomial features to a CSV file
        pd.DataFrame(preprocessed_data_poly).to_csv(f'{symbol}_preprocessed_poly.csv')

        # Encrypt the preprocessed data
        with open(f'{symbol}_preprocessed.csv', 'rb') as file:
            preprocessed_data = file.read()
            encrypted_preprocessed_data = cipher_suite.encrypt(preprocessed_data)
        with open(f'encrypted_{symbol}_preprocessed.csv', 'wb') as file:
            file.write(encrypted_preprocessed_data)

        # Decrypt the preprocessed data
        with open(f'encrypted_{symbol}_preprocessed.csv', 'rb') as file:  # Corrected file name
            encrypted_preprocessed_data = file.read()
            decrypted_preprocessed_data = cipher_suite.decrypt(encrypted_preprocessed_data)

            # Save the decrypted data to a new file
            with open(f'decrypted_{symbol}_preprocessed.csv', 'wb') as decrypted_file:
                decrypted_file.write(decrypted_preprocessed_data)

        # Load the data (btc)
        data = pd.read_csv('BTC-USD_preprocessed.csv')

        # Load the data (eth)
        eth_data = pd.read_csv('ETH-USD_preprocessed.csv')

        # Convert 'Date' to datetime (eth)
        eth_data['Date'] = pd.to_datetime(eth_data['Date'])

        # Convert 'Date' to the number of days since the earliest date in the dataset (eth)
        eth_data['Date'] = (eth_data['Date'] - eth_data['Date'].min()).dt.days

        # Convert 'Date' to datetime (btc)
        data['Date'] = pd.to_datetime(data['Date'])

        # Convert 'Date' to the number of days since the earliest date in the dataset (btc)
        data['Date'] = (data['Date'] - data['Date'].min()).dt.days

        # Shift the 'Close' price to predict the price three weeks ahead (eth)
        eth_data['Close_Future'] = eth_data['Close'].shift(-21)  # Predict 3 weeks ahead

        # Remove rows with NaN values in the shifted target column (eth)
        eth_data = eth_data.dropna(subset=['Close_Future'])

        # Drop the original 'Close' column as it's not a feature anymore (eth)
        X_eth = eth_data.drop(['Close', 'Close_Future'], axis=1)
        y_eth = eth_data['Close_Future']

        # Shift the 'Close' price to predict the price three weeks ahead (btc)
        data['Close_Future'] = data['Close'].shift(-21)  # Shift 21 days into the future

        # Remove rows with NaN values in the shifted target column (btc)
        data = data.dropna(subset=['Close_Future'])

        # Drop the original 'Close' column as it's not a feature anymore (btc)
        X = data.drop(['Close', 'Close_Future'], axis=1)
        y = data['Close_Future']

        # Split the data into training and test sets (btc)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Split the data into training and test sets (eth)
        X_train_eth, X_test_eth, y_train_eth, y_test_eth = train_test_split(X_eth, y_eth, test_size=0.2,
                                                                            random_state=42)

        # Standardise the features after splitting
        scaler = StandardScaler()
        X_train_eth_scaled = scaler.fit_transform(X_train_eth)
        X_test_eth_scaled = scaler.transform(X_test_eth)

        # Cross-Validation and RFE
        lasso_cv_eth = LassoCV(cv=5, random_state=42, max_iter=10000)
        selector_eth = RFE(lasso_cv_eth, n_features_to_select=5, step=1)
        selector_eth.fit(X_train_eth_scaled, y_train_eth)  # Fit RFE

        # Fit RFE on the training data and transform training and test sets
        X_train_selected_eth = selector_eth.transform(X_train_eth_scaled)  # Transform training set
        X_test_selected_eth = selector_eth.transform(X_test_eth_scaled)  # Transform test set

        # Train the Lasso model with the selected features
        lasso_cv_eth.fit(X_train_selected_eth, y_train_eth)

        # Prediction using Lasso Regression
        lasso_y_pred_eth = lasso_cv_eth.predict(X_test_selected_eth)

        # Standardise the features before applying Lasso and RFE (btc)
        scaler_btc = StandardScaler()
        X_train_btc_scaled = scaler_btc.fit_transform(X_train)
        X_test_btc_scaled = scaler_btc.transform(X_test)

        # Cross-Validation and RFE (btc)
        lasso_cv_btc = LassoCV(cv=5, random_state=42, max_iter=10000)
        selector_btc = RFE(lasso_cv_btc, n_features_to_select=5, step=1)
        selector_btc.fit(X_train_btc_scaled, y_train)

        # Fit RFE on the training data and transform training and test sets (btc)
        X_train_selected_btc = selector_btc.transform(X_train_btc_scaled)  # Transform training set
        X_test_selected_btc = selector_btc.transform(X_test_btc_scaled)  # Transform test set

        # Train the Lasso model with the selected features (btc)
        lasso_cv_btc.fit(X_train_selected_btc, y_train)

        # Prediction using Lasso Regression (btc)
        lasso_y_pred_btc = lasso_cv_btc.predict(X_test_selected_btc)

        # Train the Random Forest model for BTC-USD
        rf_model_btc = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_btc.fit(X_train, y_train)

        # Train the Random Forest model for ETH-USD
        rf_model_eth = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model_eth.fit(X_train_eth, y_train_eth)

        # Prediction for BTC-USD
        rf_y_pred_btc = rf_model_btc.predict(X_test)

        # Prediction for ETH-USD
        rf_y_pred_eth = rf_model_eth.predict(X_test_eth)

        # Calculate metrics for Lasso Regression (btc)
        lasso_mse_btc = mean_squared_error(y_test, lasso_y_pred_btc)
        lasso_mae_btc = mean_absolute_error(y_test, lasso_y_pred_btc)
        lasso_r2_btc = r2_score(y_test, lasso_y_pred_btc)

        # Prints Metrics for Lasso Regression (btc)
        print("\nLasso Regression Metrics (BTC-USD):")
        print(f'MSE: {lasso_mse_btc}')
        print(f'MAE: {lasso_mae_btc}')
        print(f'R^2: {lasso_r2_btc}')

        # Use the last 21 days of data to predict the closing price three weeks ahead
        last_21_days_btc = data[-21:].drop(['Close', 'Close_Future'], axis=1)
        last_21_days_btc_scaled = scaler_btc.transform(last_21_days_btc)  # Scale the last 21 days
        last_21_days_btc_selected = selector_btc.transform(last_21_days_btc_scaled)  # Select features

        # Output the predicted closing price for three weeks ahead
        future_prediction_btc = lasso_cv_btc.predict(last_21_days_btc_selected)
        print(f"Predicted closing price for BTC three weeks ahead (Lasso Regression): {future_prediction_btc[-1]}")

        # Calculate metrics for Lasso Regression (eth)
        lasso_mse_eth = mean_squared_error(y_test_eth, lasso_y_pred_eth)
        lasso_mae_eth = mean_absolute_error(y_test_eth, lasso_y_pred_eth)
        lasso_r2_eth = r2_score(y_test_eth, lasso_y_pred_eth)

        # Prints Metrics of Lasso Regression (eth)
        print("\nLasso Regression Metrics (ETH-USD):")
        print(f'MSE: {lasso_mse_eth}')
        print(f'MAE: {lasso_mae_eth}')
        print(f'R^2: {lasso_r2_eth}')

        # Use the last 21 days of data to predict the closing price three weeks ahead
        last_21_days_eth = eth_data[-21:].drop(['Close', 'Close_Future'], axis=1)
        last_21_days_eth_scaled = scaler.transform(last_21_days_eth)  # Scale the last 21 days
        last_21_days_eth_selected = selector_eth.transform(last_21_days_eth_scaled)  # Select features

        # Output the predicted closing price for three weeks ahead
        future_prediction_eth = lasso_cv_eth.predict(last_21_days_eth_selected)
        print(f"Predicted closing price for ETH three weeks ahead (Lasso Regression): {future_prediction_eth[-1]}")

        # Calculate metrics for Random Forest (btc)
        rf_mse = mean_squared_error(y_test, rf_y_pred_btc)
        rf_mae = mean_absolute_error(y_test, rf_y_pred_btc)
        rf_r2 = r2_score(y_test, rf_y_pred_btc)

        # Prints Metrics of Random Forest (btc)
        print("\nRandom Forest Metrics (Btc-Usd):")
        print(f'MSE: {rf_mse}')
        print(f'MAE: {rf_mae}')
        print(f'R^2: {rf_r2}')

        # Use the last 21 days of data to predict the closing price three weeks ahead
        last_21_days = data[-21:].drop(['Close', 'Close_Future'], axis=1)
        future_prediction = rf_model_btc.predict(last_21_days)

        # Output the predicted closing price for three weeks ahead
        print(f"Predicted closing price for BTC three weeks ahead (Random Forest): {future_prediction[-1]}")

        # Calculate metrics for Random Forest (eth)
        rf_mse_eth = mean_squared_error(y_test_eth, rf_y_pred_eth)
        rf_mae_eth = mean_absolute_error(y_test_eth, rf_y_pred_eth)
        rf_r2_eth = r2_score(y_test_eth, rf_y_pred_eth)

        # Prints Metrics of Random Forest (eth)
        print("\nRandom Forest Metrics (Eth-Usd):")
        print(f'MSE: {rf_mse_eth}')
        print(f'MAE: {rf_mae_eth}')
        print(f'R^2: {rf_r2_eth}')

        # Use the last 21 days of data to predict the closing price three weeks ahead
        last_21_days = data[-21:].drop(['Close', 'Close_Future'], axis=1)
        future_prediction_eth = rf_model_eth.predict(last_21_days)

        # Output the predicted closing price for three weeks ahead
        print(f"Predicted closing price for ETH three weeks ahead (Random Forest): {future_prediction_eth[-1]}")

        # Train the Linear Regression (btc)
        lr_model_btc = LinearRegression()
        lr_model_btc.fit(X_train, y_train)
        lr_y_pred_btc = lr_model_btc.predict(X_test)

        # Standardise the features after splitting (eth)
        scaler_eth = StandardScaler()
        X_train_eth_scaled = scaler_eth.fit_transform(X_train_eth)
        X_test_eth_scaled = scaler_eth.transform(X_test_eth)

        # Initialise the Recursive Feature Elimination (RFE) method with LassoCV as the estimator.
        # LassoCV is used to select the best features based on the regularisation path.
        selector_eth = RFE(lasso_cv_eth, n_features_to_select=5, step=1)
        selector_eth.fit(X_train_eth_scaled, y_train_eth)
        X_train_selected_eth = selector_eth.transform(X_train_eth_scaled)
        X_test_selected_eth = selector_eth.transform(X_test_eth_scaled)

        # Train the Linear Regression model with the selected features (eth)
        lr_model_eth = LinearRegression()
        lr_model_eth.fit(X_train_selected_eth, y_train_eth)

        # Prediction using Linear Regression (eth)
        lr_y_pred_eth = lr_model_eth.predict(X_test_selected_eth)

        # Calculate metrics for Linear Regression (btc)
        lr_mse = mean_squared_error(y_test, lr_y_pred_btc)
        lr_mae = mean_absolute_error(y_test, lr_y_pred_btc)
        lr_r2 = r2_score(y_test, lr_y_pred_btc)

        # Prints Metrics of Linear Regression (btc)
        print("\nLinear Regression Metrics (Btc-Usd):")
        print(f'MSE: {lr_mse}')
        print(f'MAE: {lr_mae}')
        print(f'R^2: {lr_r2}')

        # Use the last 21 days of data to predict the closing price three weeks ahead
        last_21_days = data[-21:].drop(['Close', 'Close_Future'], axis=1)
        future_prediction = lr_model_btc.predict(last_21_days)

        # Output the predicted closing price for three weeks ahead
        print(f"Predicted closing price for BTC three weeks ahead (Linear Regression): {future_prediction[-1]}")

        # Calculate metrics for Linear Regression (eth)
        lr_mse_eth = mean_squared_error(y_test_eth, lr_y_pred_eth)
        lr_mae_eth = mean_absolute_error(y_test_eth, lr_y_pred_eth)
        lr_r2_eth = r2_score(y_test_eth, lr_y_pred_eth)

        # Prints Metrics of Linear Regression (eth)
        print("\nLinear Regression Metrics (Eth-Usd):")
        print(f'MSE: {lr_mse_eth}')
        print(f'MAE: {lr_mae_eth}')
        print(f'R^2: {lr_r2_eth}')

        # Prepare the last 21 days of data for prediction (eth)
        last_21_days_eth = eth_data[-21:].drop(['Close', 'Close_Future'], axis=1)
        last_21_days_eth_scaled = scaler_eth.transform(last_21_days_eth)  # Scale the last 21 days

        # Apply the same RFE selector to the last 21 days of data
        last_21_days_eth_selected = selector_eth.transform(last_21_days_eth_scaled)
        future_prediction_eth = lr_model_eth.predict(last_21_days_eth_selected)
        print(f"Predicted closing price for ETH three weeks ahead (Linear Regression): {future_prediction_eth[-1]}")

        # Scatter plot Lasso Regression (btc)
        plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Closing Prices')
        plt.scatter(range(len(lasso_y_pred_btc)), lasso_y_pred_btc, color='blue', label='Predicted Closing Prices')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs Predicted for BTC-USD (Lasso Regression)')
        plt.legend()
        plt.show()
        plt.close()

        # Scatter plot Lasso Regression (eth)
        plt.scatter(range(len(y_test_eth)), y_test_eth, color='red', label='Actual Closing Prices')
        plt.scatter(range(len(lasso_y_pred_eth)), lasso_y_pred_eth, color='blue', label='Predicted Closing Prices')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs Predicted for ETH-USD (Lasso Regression)')
        plt.legend()
        plt.show()
        plt.close()

        # Line graph Lasso Regression (btc)
        plt.plot(y_test.reset_index(drop=True), label='Actual Values')
        plt.plot(lasso_y_pred_btc, label='Predicted Values')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs. Predicted for BTC-USD (Lasso Regression)')
        plt.legend()
        plt.show()
        plt.close()

        # Line graph Lasso Regression (eth)
        plt.plot(y_test_eth.reset_index(drop=True), label='Actual Values')
        plt.plot(lasso_y_pred_eth, label='Predicted Values')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs. Predicted for ETH-USD (Lasso Regression)')
        plt.legend()
        plt.show()
        plt.close()

        # Scatter plot Random Forest (btc)
        plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Closing Prices')
        plt.scatter(range(len(rf_y_pred_btc)), rf_y_pred_btc, color='blue', label='Predicted Closing Prices')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs Predicted for BTC-USD (Random Forest Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Scatter plot Random Forest (eth)
        plt.scatter(range(len(y_test_eth)), y_test_eth, color='red', label='Actual Closing Prices')
        plt.scatter(range(len(rf_y_pred_eth)), rf_y_pred_eth, color='blue', label='Predicted Closing Prices')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs Predicted for ETH-USD (Random Forest Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Line graph Random Forest (btc)
        plt.plot(y_test.reset_index(drop=True), label='Actual Values')
        plt.plot(rf_y_pred_btc, label='Predicted Values')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs. Predicted for BTC-USD (Random Forest Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Line graph Random Forest (eth)
        plt.plot(y_test_eth.reset_index(drop=True), label='Actual Values')
        plt.plot(rf_y_pred_eth, label='Predicted Values')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs. Predicted for ETH-USD (Random Forest Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Scatter plot Linear Regression (btc)
        plt.scatter(range(len(y_test)), y_test, color='red', label='Actual Closing Prices')
        plt.scatter(range(len(lr_y_pred_btc)), lr_y_pred_btc, color='blue', label='Predicted Closing Prices')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing  Prices')
        plt.title('Actual vs. Predicted for BTC-USD (Linear Regression Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Scatter plot Linear Regression (eth)
        plt.scatter(range(len(y_test_eth)), y_test_eth, color='red', label='Actual Closing Prices')
        plt.scatter(range(len(lr_y_pred_eth)), lr_y_pred_eth, color='blue', label='Predicted Closing Prices')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs. Predicted for ETH-USD (Linear Regression Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Line graph Linear Regression (btc)
        plt.plot(y_test.reset_index(drop=True), label='Actual Values')
        plt.plot(lr_y_pred_btc, label='Predicted Values')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs. Predicted for BTC-USD (Linear Regression Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Line graph Linear Regression (eth)
        plt.plot(y_test_eth.reset_index(drop=True), label='Actual Values')
        plt.plot(lr_y_pred_eth, label='Predicted Values')
        plt.xlabel('Index from test dataset')
        plt.ylabel('Closing Prices')
        plt.title('Actual vs. Predicted for ETH-USD (Linear Regression Model)')
        plt.legend()
        plt.show()
        plt.close()

        # Prediction Graph (BTC-USD)
        plt.figure(figsize=(12, 6))
        plt.plot(lasso_y_pred_btc, label='Lasso Regression')
        plt.plot(rf_y_pred_btc, label='Random Forest')
        plt.plot(lr_y_pred_btc, label='Linear Regression')
        plt.xlabel('Index')
        plt.ylabel('Predicted Closing Price')
        plt.title('Predicted Closing Prices (BTC-USD)')
        plt.legend()
        plt.show()

        # Create a range of days from 1 to 21
        days = range(1, 22)

        # 21-day Prediction Graph (BTC-USD)
        plt.figure(figsize=(12, 6))
        plt.plot(days, lasso_cv_btc.predict(last_21_days_btc_selected), label='Lasso Regression')
        plt.plot(days, rf_model_btc.predict(data[-21:].drop(['Close', 'Close_Future'], axis=1)), label='Random Forest')
        plt.plot(days, lr_model_btc.predict(data[-21:].drop(['Close', 'Close_Future'], axis=1)),
                 label='Linear Regression')
        plt.xlabel('Days')
        plt.ylabel('Predicted Closing Price')
        plt.title('21-day Predictions (BTC-USD)')
        plt.legend()
        plt.xticks(days)  # Set x-axis ticks to show days from 1 to 21
        plt.show()

        # 21-day Prediction Graph (ETH-USD)
        plt.figure(figsize=(12, 6))
        plt.plot(days, lasso_cv_eth.predict(last_21_days_eth_selected), label='Lasso Regression')
        plt.plot(days, rf_model_eth.predict(eth_data[-21:].drop(['Close', 'Close_Future'], axis=1)),
                 label='Random Forest')
        plt.plot(days, lr_model_eth.predict(last_21_days_eth_selected), label='Linear Regression')
        plt.xlabel('Days')
        plt.ylabel('Predicted Closing Price')
        plt.title('21-day Predictions (ETH-USD)')
        plt.legend()
        plt.xticks(days)  # Set x-axis ticks to show days from 1 to 21
        plt.show()

        # Define the date range
        start_date = '2024-03-18'
        end_date = '2024-04-09'

        # Download historical data for Ethereum and Bitcoin
        eth_data = yf.download('ETH-USD', start=start_date, end=end_date)
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)

        # Create a range of numbers representing each day in the period
        days = range(len(eth_data))

        # Plot the actual closing prices for Ethereum with days on the x-axis
        plt.figure(figsize=(12, 6))
        plt.plot(days, eth_data['Close'], label='Actual ETH-USD')
        plt.xlabel('Days')
        plt.ylabel('Closing Price')
        plt.title('Actual Closing Prices (ETH-USD)')
        plt.legend()
        plt.xticks(days)  # Set x-axis ticks to show days
        plt.show()

        # Plot the actual closing prices for Bitcoin with days on the x-axis
        plt.figure(figsize=(12, 6))
        plt.plot(days, btc_data['Close'], label='Actual BTC-USD')
        plt.xlabel('Days')
        plt.ylabel('Closing Price')
        plt.title('Actual Closing Prices (BTC-USD)')
        plt.legend()
        plt.xticks(days)  # Set x-axis ticks to show days
        plt.show()

        # Prediction Graph (ETH-USD)
        plt.figure(figsize=(12, 6))
        plt.plot(lasso_y_pred_eth, label='Lasso Regression')
        plt.plot(rf_y_pred_eth, label='Random Forest')
        plt.plot(lr_y_pred_eth, label='Linear Regression')
        plt.xlabel('Index')
        plt.ylabel('Predicted Closing Price')
        plt.title('Predicted Closing Prices (ETH-USD)')
        plt.legend()
        plt.show()

        # Define the date range
        start_date = '2024-03-18'
        end_date = '2024-04-09'

        # Download historical data for Ethereum and Bitcoin
        eth_data = yf.download('ETH-USD', start=start_date, end=end_date)
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date)

        # Placeholders for the predictions
        btc_predictions_lasso = lasso_cv_btc.predict(last_21_days_btc_selected)   # Placeholder for BTC predictions using Lasso
        btc_predictions_rf = rf_model_btc.predict(last_21_days)  # Placeholder for BTC predictions using Random Forest
        btc_predictions_lr = lr_model_btc.predict(last_21_days)  # Placeholder for BTC predictions using Linear Regression

        eth_predictions_lasso = lasso_cv_eth.predict(last_21_days_eth_selected)  # Placeholder for ETH predictions using Lasso
        eth_predictions_rf = rf_model_eth.predict(last_21_days)  # Placeholder for ETH predictions using Random Forest
        eth_predictions_lr = lr_model_eth.predict(last_21_days_eth_selected)  # Placeholder for ETH predictions using Linear Regression

        # Create a DataFrame to hold the actual and predicted prices (btc)
        btc_results = pd.DataFrame({
            'Day': range(1, 22),
            'Actual_BTC': btc_data['Close'].values[-21:],
            'Predicted_BTC_Lasso': btc_predictions_lasso,
            'Predicted_BTC_RF': btc_predictions_rf,
            'Predicted_BTC_LR': btc_predictions_lr
        })

        # Create a DataFrame to hold the actual and predicted prices (eth)
        eth_results = pd.DataFrame({
            'Day': range(1, 22),
            'Actual_ETH': eth_data['Close'].values[-21:],
            'Predicted_ETH_Lasso': eth_predictions_lasso,
            'Predicted_ETH_RF': eth_predictions_rf,
            'Predicted_ETH_LR': eth_predictions_lr
        })

        # Function to calculate MAPE
        def calculate_mape(actual, predicted):
            return np.mean(np.abs((actual - predicted) / actual)) * 100

        # Function to calculate accuracy
        def calculate_accuracy(mape):
            return 100 - mape

        # Calculate MAPE for each model
        btc_results['MAPE_BTC_Lasso'] = calculate_mape(btc_results['Actual_BTC'], btc_results['Predicted_BTC_Lasso'])
        btc_results['MAPE_BTC_RF'] = calculate_mape(btc_results['Actual_BTC'], btc_results['Predicted_BTC_RF'])
        btc_results['MAPE_BTC_LR'] = calculate_mape(btc_results['Actual_BTC'], btc_results['Predicted_BTC_LR'])

        eth_results['MAPE_ETH_Lasso'] = calculate_mape(eth_results['Actual_ETH'], eth_results['Predicted_ETH_Lasso'])
        eth_results['MAPE_ETH_RF'] = calculate_mape(eth_results['Actual_ETH'], eth_results['Predicted_ETH_RF'])
        eth_results['MAPE_ETH_LR'] = calculate_mape(eth_results['Actual_ETH'], eth_results['Predicted_ETH_LR'])

        # Calculate overall accuracy for each model
        btc_results['Accuracy_BTC_Lasso'] = calculate_accuracy(btc_results['MAPE_BTC_Lasso'])
        btc_results['Accuracy_BTC_RF'] = calculate_accuracy(btc_results['MAPE_BTC_RF'])
        btc_results['Accuracy_BTC_LR'] = calculate_accuracy(btc_results['MAPE_BTC_LR'])

        eth_results['Accuracy_ETH_Lasso'] = calculate_accuracy(eth_results['MAPE_ETH_Lasso'])
        eth_results['Accuracy_ETH_RF'] = calculate_accuracy(eth_results['MAPE_ETH_RF'])
        eth_results['Accuracy_ETH_LR'] = calculate_accuracy(eth_results['MAPE_ETH_LR'])

        # Calculate the overall accuracy percentage as the mean of individual accuracies for each model
        btc_overall_accuracy_lasso = btc_results['Accuracy_BTC_Lasso'].mean()
        btc_overall_accuracy_rf = btc_results['Accuracy_BTC_RF'].mean()
        btc_overall_accuracy_lr = btc_results['Accuracy_BTC_LR'].mean()

        eth_overall_accuracy_lasso = eth_results['Accuracy_ETH_Lasso'].mean()
        eth_overall_accuracy_rf = eth_results['Accuracy_ETH_RF'].mean()
        eth_overall_accuracy_lr = eth_results['Accuracy_ETH_LR'].mean()

        # Print the overall accuracy percentage for each model (btc)
        print("Overall Accuracy for BTC-USD Predictions:")
        print(f"Lasso Regression: {btc_overall_accuracy_lasso:.2f}%")
        print(f"Random Forest: {btc_overall_accuracy_rf:.2f}%")
        print(f"Linear Regression: {btc_overall_accuracy_lr:.2f}%")

        # Print the overall accuracy percentage for each model (eth)
        print("\nOverall Accuracy for ETH-USD Predictions:")
        print(f"Lasso Regression: {eth_overall_accuracy_lasso:.2f}%")
        print(f"Random Forest: {eth_overall_accuracy_rf:.2f}%")
        print(f"Linear Regression: {eth_overall_accuracy_lr:.2f}%")

        # Output the results to CSV files
        btc_results.to_csv('btc_predictions_vs_actual.csv', index=False)
        eth_results.to_csv('eth_predictions_vs_actual.csv', index=False)

        # Print overall MAPE for each model (btc)
        print("Overall MAPE for BTC-USD Predictions:")
        print("Lasso Regression:", btc_results['MAPE_BTC_Lasso'].mean())
        print("Random Forest:", btc_results['MAPE_BTC_RF'].mean())
        print("Linear Regression:", btc_results['MAPE_BTC_LR'].mean())

        # Print overall MAPE for each model (eth)
        print("\nOverall MAPE for ETH-USD Predictions:")
        print("Lasso Regression:", eth_results['MAPE_ETH_Lasso'].mean())
        print("Random Forest:", eth_results['MAPE_ETH_RF'].mean())
        print("Linear Regression:", eth_results['MAPE_ETH_LR'].mean())

        # Reset the index of y_test for proper alignment with predictions
        y_test_eth_reset = y_test.reset_index(drop=True)

        # Calculate the absolute errors
        absolute_errors = abs(y_test_eth - rf_y_pred_eth)

        # Calculate the Mean Absolute Percentage Error (MAPE) for each prediction
        mape = (absolute_errors / y_test_eth) * 100

        # Calculate the accuracy for each prediction as (100 - MAPE)
        accuracy = 100 - mape

        # Add the absolute errors and accuracy to the comparison DataFrame
        comparison_df = pd.DataFrame({
            'Actual Values': y_test_eth,
            'Predicted Values': rf_y_pred_eth,
            'Absolute Error': absolute_errors,
            'Accuracy (%)': accuracy
        })
        # Display the DataFrame with 'Accuracy (%)' column
        print(comparison_df)

        # Calculate the overall accuracy percentage as the mean of individual accuracies
        overall_accuracy = accuracy.mean()

        # Print the overall accuracy percentage
        print(f"Overall Accuracy Percentage: {overall_accuracy:.2f}%")

        # Save the DataFrame with the accuracy column to a CSV file
        comparison_df.to_csv('ETH-USD_Actual_vs_Predicted_with_Accuracy.csv', index=False)

        # Reset the index of y_test_eth for proper alignment with predictions
        y_test_eth_reset = y_test_eth.reset_index(drop=True)

        # Calculate the absolute errors for each model
        lasso_absolute_errors_eth = abs(y_test_eth_reset - lasso_y_pred_eth)
        rf_absolute_errors_eth = abs(y_test_eth_reset - rf_y_pred_eth)
        lr_absolute_errors_eth = abs(y_test_eth_reset - lr_y_pred_eth)

        # Calculate the Mean Absolute Percentage Error (MAPE) for each model
        lasso_mape_eth = (lasso_absolute_errors_eth / y_test_eth_reset) * 100
        rf_mape_eth = (rf_absolute_errors_eth / y_test_eth_reset) * 100
        lr_mape_eth = (lr_absolute_errors_eth / y_test_eth_reset) * 100

        # Calculate the accuracy for each prediction as (100 - MAPE)
        lasso_accuracy_eth = 100 - lasso_mape_eth
        rf_accuracy_eth = 100 - rf_mape_eth
        lr_accuracy_eth = 100 - lr_mape_eth

        # Add the absolute errors and accuracy to the comparison DataFrame
        comparison_df_eth = pd.DataFrame({
            'Actual Values': y_test_eth_reset,
            'Lasso Predicted Values': lasso_y_pred_eth,
            'Random Forest Predicted Values': rf_y_pred_eth,
            'Linear Regression Predicted Values': lr_y_pred_eth,
            'Lasso Absolute Error': lasso_absolute_errors_eth,
            'Random Forest Absolute Error': rf_absolute_errors_eth,
            'Linear Regression Absolute Error': lr_absolute_errors_eth,
            'Lasso Accuracy (%)': lasso_accuracy_eth,
            'Random Forest Accuracy (%)': rf_accuracy_eth,
            'Linear Regression Accuracy (%)': lr_accuracy_eth
        })

        # Display the DataFrame with the 'Accuracy (%)' column
        print(comparison_df_eth)

        # Calculate the overall accuracy percentage as the mean of individual accuracies for each model
        lasso_overall_accuracy_eth = lasso_accuracy_eth.mean()
        rf_overall_accuracy_eth = rf_accuracy_eth.mean()
        lr_overall_accuracy_eth = lr_accuracy_eth.mean()

        # Print the overall accuracy percentage for each model
        print(f"Lasso Regression Overall Accuracy Percentage: {lasso_overall_accuracy_eth:.2f}%")
        print(f"Random Forest Overall Accuracy Percentage: {rf_overall_accuracy_eth:.2f}%")
        print(f"Linear Regression Overall Accuracy Percentage: {lr_overall_accuracy_eth:.2f}%")

        # Save the DataFrame with the accuracy column to a CSV file
        comparison_df_eth.to_csv('ETH-USD_Actual_vs_Predicted_Predicted_Closing_Prices.csv', index=False)

        # Initialise the Binance client
        exchange = ccxt.binance({
            'apiKey': '2eEqIn1zHBS42RwMjFdQaUNVctCtICTesovUf59oOiV3ffg0r3XVH6e3sI3Jwm1Q',
            'secret': 'Xeq9r1uzwTnsRQTlG2csAPmi92zNYQuS3ESXSls4wOkoAOgimJF0Gm8d91mpHZaQ',
        })

        def update_plot(timeframe):
            symbols = ['BTC/USDT', 'ETH/USDT']
            # Separate figures for each symbol
            figs = {sym: plt.figure() for sym in symbols}
            axs = {sym: figs[sym].add_subplot(1, 1, 1) for sym in symbols}
            timestamps = {sym: [] for sym in symbols}
            closing_prices = {sym: [] for sym in symbols}

            # This function animates the live cryptocurrency prices for specified symbols
            def animate(_):
                for sym in symbols:
                    axs[sym].clear()
                    data = exchange.fetch_ohlcv(sym, timeframe)
                    if data:
                        last_timestamp = timestamps[sym][-1] if timestamps[sym] else 0
                        new_data = [candle for candle in data if candle[0] / 1000 > last_timestamp]
                        timestamps[sym] += [datetime.fromtimestamp(candle[0] / 1000) for candle in new_data]
                        closing_prices[sym] += [candle[4] for candle in new_data]
                        axs[sym].plot(timestamps[sym], closing_prices[sym], label=sym)

                        # Increase figure size
                        figs[sym].set_size_inches(12, 8)

                        # Creates graph for live data
                        axs[sym].set_xlabel('Time')
                        axs[sym].set_ylabel('Price (USDT)')
                        axs[sym].set_title(f'Live Cryptocurrency Prices for {sym}')
                        axs[sym].legend()

                        plt.draw()
                        plt.pause(0.001)

            # Creates an animation that updates the plot for the first symbol in the 'symbols' list
            ani = animation.FuncAnimation(figs[symbols[0]], animate, interval=10000, cache_frame_data=False,
                                          save_count=100)
            plt.show()


        update_plot('1m')

        # Wait for 24 hours before fetching the data again
    time.sleep(86400)  # 86400 seconds = 24 hours

