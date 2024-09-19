
# Stock Price Prediction Dashboard

This project is a **Dash-based web application** for predicting stock prices using SARIMAX and XGBoost models. The application allows users to upload stock price data, visualize the predicted prices, and review key model evaluation metrics.

## Sentimental Analysis

This project uses the **Moving Average Convergence Divergence (MACD)** and the **Relative Strength Index (RSI)** as sentiment analysis tools to better understand price trends and momentum. These technical indicators help inform the models of possible price movements and market sentiment.

- **MACD**: Helps to identify changes in the strength, direction, momentum, and duration of a trend.
- **RSI**: Measures the speed and change of price movements, indicating overbought or oversold conditions.

## Models Used

Two models are used in this project:
1. **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors):
   - SARIMAX is a popular time series forecasting method that accounts for seasonality and external factors (exogenous variables).
   - It is used to predict future stock prices based on historical data and trends.
2. **XGBoost** (Extreme Gradient Boosting):
   - XGBoost is a powerful machine learning algorithm that is widely used for regression and classification tasks.
   - It is used here to predict the next day’s stock prices using numerical features from the dataset.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-prediction-dashboard.git
   cd stock-prediction-dashboard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   python app.py
   ```

The app will be available at `http://127.0.0.1:8056/`.

## Usage

1. **Upload CSV**: You can upload stock data (in CSV format), and the app will process it to predict future prices using SARIMAX and XGBoost models.
2. **Select Date Range**: Filter the prediction results by selecting a specific date range.
3. **Visualizations**: The app provides the following visualizations:
   - Actual vs Predicted Prices using SARIMAX
   - Actual vs Predicted Prices using XGBoost
   - MACD and MACD Signal Line
   - Relative Strength Index (RSI)

4. **Model Evaluation**: The app displays key metrics like MSE (Mean Squared Error), MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and R² for both models.

## How It Helps People

This dashboard provides traders and investors with the ability to:
- **Visualize stock price trends** and predict future price movements.
- Use sentiment indicators like **MACD** and **RSI** to understand market conditions.
- Compare the accuracy of two powerful forecasting models (SARIMAX and XGBoost) to make informed investment decisions.

## Limitations

While the dashboard provides useful forecasts, it has several limitations:
- **Limited to historical data**: The models depend solely on historical stock prices and trends. External factors like news, market shocks, and political events are not considered.
- **Generalization**: The accuracy of the models may degrade if the stock prices exhibit sudden, unexpected changes that weren't captured in the training data.
- **Model Assumptions**: The models assume that the future will somewhat resemble the past. This may not always hold true, especially during volatile market periods.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
