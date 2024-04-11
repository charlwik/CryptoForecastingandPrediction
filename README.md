The provided code is a comprehensive script designed for cryptocurrency price prediction and visualisation, focusing on Bitcoin (BTC) and Ethereum (ETH). It encompasses several stages of data processing, model training, prediction, and visualization, aiming to forecast future cryptocurrency prices based on historical data. Here's an expanded overview of the key components and functionalities within the script:

Data Retrieval: The script uses the yfinance library to fetch historical price data for BTC and ETH from Yahoo Finance. This data includes open, high, low, close prices, and volume, which are crucial for analysing market trends.

Data Preprocessing:

Cleaning: Removes rows with missing values to ensure data quality.
Scaling: Applies Min-Max scaling to numerical features, ensuring they contribute equally to the analysis.
Feature Engineering: Calculates a 50-day moving average and adds polynomial features to capture non-linear relationships. It also includes domain-specific features like 'Price Change' (Close - Open).
Feature Selection: Utilises Recursive Feature Elimination (RFE) with Lasso Cross-Validation (LassoCV) to identify and select the most relevant features for the prediction models. This step is crucial for improving model performance by eliminating redundant or less informative features.
Model Training and Prediction:
Trains Lasso Regression, Random Forest, and Linear Regression models on the preprocessed and feature-selected data.
Makes predictions on the test set for both BTC and ETH, aiming to forecast future prices three weeks ahead.
Performance Evaluation: Calculates and prints metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R^2) to evaluate the performance of each model.

Visualisation:

Generates scatter plots and line graphs to compare actual vs. predicted prices, providing a visual assessment of model accuracy.
Includes a unique visualisation feature that uses matplotlib.animation to display live cryptocurrency prices fetched from the Binance exchange, offering real-time market insights.
Encryption and Decryption: For security purposes, the preprocessed data is encrypted using the cryptography library before being saved to disk. It is then decrypted when needed for further processing, ensuring data confidentiality.
Scheduling: The script is designed to run continuously, updating predictions daily by fetching new data and repeating the entire process. This is achieved through a while loop that pauses execution for 24 hours before restarting.

Output: 

Saves various outputs, including preprocessed data, model predictions, and performance metrics, to CSV files for further analysis or reporting.
Accuracy and Error Analysis: Calculates the Mean Absolute Percentage Error (MAPE) and overall accuracy for each model, providing insights into the predictive reliability of the models.

This script represents a comprehensive approach to cryptocurrency price forecasting, integrating data retrieval, preprocessing, modeling, and visualisation into a single automated workflow. It leverages machine learning techniques to analyze historical data and predict future market trends, offering valuable insights for investors and traders.
