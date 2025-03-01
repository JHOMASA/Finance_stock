import yfinance as yf
import pandas as pd
import numpy as np
import requests
import cohere
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# Initialize Cohere client
co = cohere.Client("YvexoWfYcfq9dxlWGt0EluWfYwfWwx5fbd6XJ4Aj")  # Replace with your Cohere API key

# Fetch stock data
def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        stock_data = stock.history(period="1y")
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Fetch news articles
def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=YOUR_NEWSAPI_KEY"  # Replace with your NewsAPI key
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json()["articles"]
            return articles
        else:
            print(f"NewsAPI error: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Analyze sentiment of news articles using Cohere
def analyze_news_sentiment(articles):
    sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        text = f"{title}. {description}"
        try:
            sentiment = co.classify(text).classifications[0].prediction
            article["sentiment"] = sentiment
            sentiment_counts[sentiment] += 1
        except Exception as e:
            print(f"Error analyzing sentiment for article: {text}. Error: {e}")
            article["sentiment"] = "ERROR"
    return sentiment_counts

# Calculate risk metrics
def calculate_risk_metrics(stock_data):
    try:
        returns = stock_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        max_drawdown = (stock_data['Close'] / stock_data['Close'].cummax() - 1).min()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe Ratio
        var_95 = np.percentile(returns, 5)  # Value at Risk (95% confidence)
        return {
            "Volatility": f"{volatility:.2%}",
            "Max Drawdown": f"{max_drawdown:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "VaR (95%)": f"{var_95:.2%}"
        }
    except Exception as e:
        print(f"Error calculating risk metrics: {e}")
        return {}

# Monte Carlo Simulation
def monte_carlo_simulation(stock_data, num_simulations=1000, days=252):
    try:
        if stock_data.empty:
            raise ValueError("No stock data available for simulation.")

        returns = stock_data['Close'].pct_change().dropna()
        if len(returns) < 2:
            raise ValueError("Insufficient data to calculate returns.")

        mu = returns.mean()
        sigma = returns.std()
        simulations = np.zeros((days, num_simulations))
        S0 = stock_data['Close'].iloc[-1]  # Last observed price

        for i in range(num_simulations):
            daily_returns = np.random.normal(mu, sigma, days)
            simulations[:, i] = S0 * (1 + daily_returns).cumprod()

        return simulations
    except Exception as e:
        print(f"Error in Monte Carlo simulation: {e}")
        return None

# Generate recommendations
def generate_recommendations(stock_data, financial_ratios, period=30):
    recommendations = []

    # Analyze stock trend
    if len(stock_data) >= period:
        trend = "Upward" if stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[-period] else "Downward"
        if trend == "Upward":
            recommendations.append(f"The stock is in an upward trend over the last {period} days. Consider holding or buying more.")
        elif trend == "Downward":
            recommendations.append(f"The stock is in a downward trend over the last {period} days. Consider selling or setting stop-loss orders.")
    else:
        recommendations.append("Insufficient data to determine the stock trend.")

    # Analyze financial ratios
    benchmarks = {
        "Volatility": "15%",
        "Max Drawdown": "20%",
        "Sharpe Ratio": "1.0",
        "VaR (95%)": "5%"
    }
    for ratio, value in financial_ratios.items():
        if ratio in benchmarks:
            recommendations.append(f"{ratio}: {value} (Benchmark: {benchmarks[ratio]})")

    return recommendations

# Chat with Cohere
def chat_with_cohere(prompt, context=None):
    try:
        if context:
            # Truncate the context to avoid exceeding token limits
            max_tokens = 3000  # Leave room for the prompt and response
            truncated_context = context[:max_tokens]
            prompt = f"{truncated_context}\n\nUser: {prompt}\nAssistant:"

        response = co.generate(
            model="command",
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
            stop_sequences=["\n"]
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print(f"Error in Cohere chat: {e}")
        return f"Sorry, I couldn't generate a response. Error: {str(e)}"

# Prepare data for LSTM
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']].values)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Train LSTM model
def train_lstm_model(data):
    X, y, scaler = prepare_lstm_data(data)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, batch_size=32, epochs=10)
    return model, scaler

# Train XGBoost model
def train_xgboost_model(data):
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Returns']].shift(1).dropna()
    y = data['Close'][1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    return model

# Train ARIMA model
def train_arima_model(data):
    model = ARIMA(data['Close'], order=(5, 1, 0))  # (p, d, q) parameters
    model_fit = model.fit()
    return model_fit

# Train Prophet model
def train_prophet_model(data):
    df = data[['Close']].reset_index()
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    return model

# Train Random Forest model
def train_random_forest_model(data):
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Returns']].shift(1).dropna()
    y = data['Close'][1:]
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model

# Train Linear Regression model
def train_linear_regression_model(data):
    data['Returns'] = data['Close'].pct_change()
    data = data.dropna()
    X = data[['Returns']].shift(1).dropna()
    y = data['Close'][1:]
    model = LinearRegression()
    model.fit(X, y)
    return model

# Predict using LSTM
def predict_lstm(model, scaler, data, look_back=60):
    last_sequence = scaler.transform(data[['Close']].values[-look_back:])
    last_sequence = np.reshape(last_sequence, (1, look_back, 1))
    predictions = []
    for _ in range(30):  # Predict next 30 days
        pred = model.predict(last_sequence)
        predictions.append(pred[0][0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[pred]], axis=1)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Predict using XGBoost
def predict_xgboost(model, data):
    last_return = data['Close'].pct_change().iloc[-1]
    predictions = []
    for _ in range(30):  # Predict next 30 days
        pred = model.predict(np.array([[last_return]]))
        predictions.append(pred[0])
        last_return = (pred[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    return predictions

# Predict using ARIMA
def predict_arima(model, steps=30):
    predictions = model.forecast(steps=steps)
    return predictions

# Predict using Prophet
def predict_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast['yhat'][-periods:].values

# Predict using Random Forest
def predict_random_forest(model, data, steps=30):
    predictions = []
    last_return = data['Close'].pct_change().iloc[-1]
    for _ in range(steps):
        pred = model.predict([[last_return]])
        predictions.append(pred[0])
        last_return = (pred[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    return predictions

# Predict using Linear Regression
def predict_linear_regression(model, data, steps=30):
    predictions = []
    last_return = data['Close'].pct_change().iloc[-1]
    for _ in range(steps):
        pred = model.predict([[last_return]])
        predictions.append(pred[0])
        last_return = (pred[0] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]
    return predictions

# Predict using Moving Average
def predict_moving_average(data, window=30):
    predictions = data['Close'].rolling(window=window).mean().iloc[-30:].values
    return predictions

# App layout
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='chat-context', data=""),  # Store chat context
    html.Div([
        html.H1("Stock Analysis Dashboard", style={'textAlign': 'center', 'color': '#4CAF50', 'marginBottom': '20px'}),
        dcc.Tabs(id='tabs', value='tab-stock', children=[
            dcc.Tab(label='Stock Analysis', value='tab-stock', style={'backgroundColor': '#f9f9f9', 'color': '#4CAF50'}, selected_style={'backgroundColor': '#4CAF50', 'color': 'white'}, children=[
                html.Div([
                    dcc.Input(id='stock-input-1', type='text', placeholder='Enter Stock Ticker', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Submit', id='submit-stock-1', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    dcc.Graph(id='stock-graph', style={'marginTop': '20px'}),
                    html.H4("Chat", style={'marginTop': '20px', 'color': '#4CAF50'}),
                    dcc.Input(id='chat-input-1', type='text', placeholder='Ask me anything...', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Send', id='send-chat-1', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='chat-response-1', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#fff'})
            ]),
            dcc.Tab(label='Monte Carlo Simulation', value='tab-montecarlo', style={'backgroundColor': '#f9f9f9', 'color': '#FF5722'}, selected_style={'backgroundColor': '#FF5722', 'color': 'white'}, children=[
                html.Div([
                    dcc.Input(id='stock-input-2', type='text', placeholder='Enter Stock Ticker', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Submit', id='submit-stock-2', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#FF5722', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    dcc.Graph(id='montecarlo-graph', style={'marginTop': '20px'}),
                    html.H4("Chat", style={'marginTop': '20px', 'color': '#FF5722'}),
                    dcc.Input(id='chat-input-2', type='text', placeholder='Ask me anything...', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Send', id='send-chat-2', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#FF5722', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='chat-response-2', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'}),
                    html.Div([
                        html.H2("AI-Powered Risk Profiling & Scenario Testing", style={'color': '#FF5722'}),
                        html.Label("Select Your Risk Level:", style={'marginTop': '10px'}),
                        dcc.Dropdown(
                            id='risk-profile',
                            options=[
                                {'label': 'Conservative', 'value': 'conservative'},
                                {'label': 'Moderate', 'value': 'moderate'},
                                {'label': 'Aggressive', 'value': 'aggressive'}
                            ],
                            value='moderate',
                            style={'margin': '10px', 'width': '200px'}
                        ),
                        html.Label("Select Market Scenario:", style={'marginTop': '10px'}),
                        dcc.RadioItems(
                            id='market-scenario',
                            options=[
                                {'label': 'Normal', 'value': 'normal'},
                                {'label': 'Crash (-30%)', 'value': 'crash'},
                                {'label': 'Recovery (+20%)', 'value': 'recovery'}
                            ],
                            value='normal',
                            style={'margin': '10px'}
                        ),
                        html.Label("Set Stock Alert Threshold ($):", style={'marginTop': '10px'}),
                        dcc.Input(id='alert-threshold', type='number', value=100, style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                        html.Button('Set Alert', id='set-alert', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#FF5722', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                        html.Div(id='simulation-output', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'}),
                        html.Div(id='alert-message', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'})
                    ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#fff'})
                ], style={'padding': '20px'})
            ]),
            dcc.Tab(label='Financial Ratios', value='tab-ratios', style={'backgroundColor': '#f9f9f9', 'color': '#2196F3'}, selected_style={'backgroundColor': '#2196F3', 'color': 'white'}, children=[
                html.Div([
                    dcc.Input(id='stock-input-3', type='text', placeholder='Enter Stock Ticker', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Submit', id='submit-stock-3', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Table(id='ratios-table', style={'marginTop': '20px'}),
                    html.H4("Chat", style={'marginTop': '20px', 'color': '#2196F3'}),
                    dcc.Input(id='chat-input-3', type='text', placeholder='Ask me anything...', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Send', id='send-chat-3', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#2196F3', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='chat-response-3', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#fff'})
            ]),
            dcc.Tab(label='News Sentiment', value='tab-sentiment', style={'backgroundColor': '#f9f9f9', 'color': '#9C27B0'}, selected_style={'backgroundColor': '#9C27B0', 'color': 'white'}, children=[
                html.Div([
                    dcc.Input(id='stock-input-4', type='text', placeholder='Enter Stock Ticker', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Submit', id='submit-stock-4', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#9C27B0', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    dcc.Graph(id='sentiment-graph', style={'marginTop': '20px'}),
                    html.H4("Chat", style={'marginTop': '20px', 'color': '#9C27B0'}),
                    dcc.Input(id='chat-input-4', type='text', placeholder='Ask me anything...', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Send', id='send-chat-4', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#9C27B0', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='chat-response-4', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#fff'})
            ]),
            dcc.Tab(label='Latest News', value='tab-news', style={'backgroundColor': '#f9f9f9', 'color': '#FF9800'}, selected_style={'backgroundColor': '#FF9800', 'color': 'white'}, children=[
                html.Div([
                    dcc.Input(id='stock-input-5', type='text', placeholder='Enter Stock Ticker', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Submit', id='submit-stock-5', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='news-content', style={'marginTop': '20px'}),
                    html.H4("Chat", style={'marginTop': '20px', 'color': '#FF9800'}),
                    dcc.Input(id='chat-input-5', type='text', placeholder='Ask me anything...', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Send', id='send-chat-5', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='chat-response-5', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#fff'})
            ]),
            dcc.Tab(label='Recommendations', value='tab-recommendations', style={'backgroundColor': '#f9f9f9', 'color': '#E91E63'}, selected_style={'backgroundColor': '#E91E63', 'color': 'white'}, children=[
                html.Div([
                    dcc.Input(id='stock-input-6', type='text', placeholder='Enter Stock Ticker', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    dcc.Input(id='period-input', type='number', placeholder='Enter Analysis Period (days)', value=30, style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Submit', id='submit-stock-6', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#E91E63', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='feedback-output', style={'marginTop': '20px'}),
                    html.H4("Chat", style={'marginTop': '20px', 'color': '#E91E63'}),
                    dcc.Input(id='chat-input-6', type='text', placeholder='Ask me anything...', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Send', id='send-chat-6', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#E91E63', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='chat-response-6', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#fff'})
            ]),
            dcc.Tab(label='Predictions', value='tab-predictions', style={'backgroundColor': '#f9f9f9', 'color': '#3F51B5'}, selected_style={'backgroundColor': '#3F51B5', 'color': 'white'}, children=[
                html.Div([
                    dcc.Input(id='stock-input-7', type='text', placeholder='Enter Stock Ticker', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    dcc.Dropdown(
                        id='model-selector',
                        options=[
                            {'label': 'LSTM', 'value': 'lstm'},
                            {'label': 'XGBoost', 'value': 'xgboost'},
                            {'label': 'ARIMA', 'value': 'arima'},
                            {'label': 'Prophet', 'value': 'prophet'},
                            {'label': 'Random Forest', 'value': 'random_forest'},
                            {'label': 'Linear Regression', 'value': 'linear_regression'},
                            {'label': 'Moving Average', 'value': 'moving_average'}
                        ],
                        value='lstm',
                        style={'margin': '10px', 'width': '200px'}
                    ),
                    html.Button('Submit', id='submit-stock-7', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#3F51B5', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    dcc.Graph(id='predictions-graph', style={'marginTop': '20px'}),
                    html.H4("Chat", style={'marginTop': '20px', 'color': '#3F51B5'}),
                    dcc.Input(id='chat-input-7', type='text', placeholder='Ask me anything...', style={'margin': '10px', 'padding': '10px', 'width': '200px', 'borderRadius': '5px', 'border': '1px solid #ddd'}),
                    html.Button('Send', id='send-chat-7', n_clicks=0, style={'margin': '10px', 'padding': '10px 20px', 'backgroundColor': '#3F51B5', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}),
                    html.Div(id='chat-response-7', style={'margin': '10px', 'padding': '10px', 'backgroundColor': '#f1f1f1', 'borderRadius': '5px'})
                ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'backgroundColor': '#fff'})
            ]),
        ])
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '20px'})
])

# Callbacks
@app.callback(
    Output('stock-graph', 'figure'),
    [Input('submit-stock-1', 'n_clicks')],
    [State('stock-input-1', 'value')]
)
def update_stock_graph(n_clicks, stock_ticker):
    if n_clicks > 0 and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
            fig.update_layout(title=f"Stock Price for {stock_ticker}", xaxis_title="Date", yaxis_title="Price")
            return fig
    return go.Figure()

@app.callback(
    Output('montecarlo-graph', 'figure'),
    [Input('submit-stock-2', 'n_clicks')],
    [State('stock-input-2', 'value')]
)
def update_montecarlo_graph(n_clicks, stock_ticker):
    if n_clicks > 0 and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            simulations = monte_carlo_simulation(stock_data)
            if simulations is not None:
                fig = go.Figure()
                for i in range(min(10, simulations.shape[1])):  # Plot first 10 simulations
                    fig.add_trace(go.Scatter(
                        x=np.arange(simulations.shape[0]),
                        y=simulations[:, i],
                        mode='lines',
                        name=f'Simulation {i+1}'
                    ))
                fig.update_layout(title="Monte Carlo Simulation", xaxis_title="Days", yaxis_title="Price")
                return fig
    return go.Figure()

@app.callback(
    Output('ratios-table', 'children'),
    [Input('submit-stock-3', 'n_clicks')],
    [State('stock-input-3', 'value')]
)
def update_ratios_table(n_clicks, stock_ticker):
    if n_clicks > 0 and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            risk_metrics = calculate_risk_metrics(stock_data)
            return [
                html.Tr([html.Th('Ratio'), html.Th('Value')]),
                *[html.Tr([html.Td(ratio), html.Td(value)]) for ratio, value in risk_metrics.items()]
            ]
    return "Enter a stock ticker and click 'Submit' to see financial ratios."

@app.callback(
    Output('sentiment-graph', 'figure'),
    [Input('submit-stock-4', 'n_clicks')],
    [State('stock-input-4', 'value')]
)
def update_sentiment_graph(n_clicks, stock_ticker):
    if n_clicks > 0 and stock_ticker:
        articles = fetch_news(stock_ticker)
        if articles:
            sentiment_counts = analyze_news_sentiment(articles)
            fig = go.Figure(data=[go.Bar(
                x=list(sentiment_counts.keys()),
                y=list(sentiment_counts.values())
            )])
            fig.update_layout(
                title="News Sentiment Analysis",
                xaxis_title="Sentiment",
                yaxis_title="Count"
            )
            return fig
    return go.Figure()

@app.callback(
    Output('news-content', 'children'),
    [Input('submit-stock-5', 'n_clicks')],
    [State('stock-input-5', 'value')]
)
def update_latest_news(n_clicks, stock_ticker):
    if n_clicks > 0 and stock_ticker:
        articles = fetch_news(stock_ticker)
        if articles:
            try:
                sentiment_counts = analyze_news_sentiment(articles)
                return [
                    html.Div([
                        html.H4(article.get('title', 'No Title Available')),
                        html.P(article.get('description', 'No Description Available')),
                        html.P(f"Sentiment: {article.get('sentiment', 'N/A')} (Score: {article.get('sentiment_score', 0.0):.2f})")
                    ]) for article in articles[:5]  # Display top 5 articles
                ]
            except Exception as e:
                print(f"Error processing articles: {e}")
                return html.Div("Error processing news articles. Please try again.")
        else:
            return html.Div("No news articles found for this stock ticker.")
    return html.Div("Enter a stock ticker and click 'Submit' to see the latest news.")

@app.callback(
    Output('feedback-output', 'children'),
    [Input('submit-stock-6', 'n_clicks')],
    [State('stock-input-6', 'value'), State('period-input', 'value')]
)
def update_recommendations(n_clicks, stock_ticker, period):
    if n_clicks > 0 and stock_ticker and period:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            financial_ratios = calculate_risk_metrics(stock_data)
            recommendations = generate_recommendations(stock_data, financial_ratios, period)
            return html.Ul([html.Li(recommendation) for recommendation in recommendations])
    return "Enter a stock ticker and analysis period, then click 'Submit' to see recommendations."

@app.callback(
    Output('predictions-graph', 'figure'),
    [Input('submit-stock-7', 'n_clicks')],
    [State('stock-input-7', 'value'), State('model-selector', 'value')]
)
def update_predictions(n_clicks, stock_ticker, model_type):
    if n_clicks > 0 and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            try:
                if model_type == 'lstm':
                    print("Training LSTM model...")
                    if len(stock_data) < 60:
                        print("Error: Insufficient data for LSTM (requires at least 60 days).")
                        return go.Figure()
                    X, y, scaler = prepare_lstm_data(stock_data)
                    model, _ = train_lstm_model(stock_data)
                    predictions = predict_lstm(model, scaler, stock_data)

                elif model_type == 'xgboost':
                    print("Training XGBoost model...")
                    model = train_xgboost_model(stock_data)
                    predictions = predict_xgboost(model, stock_data)

                elif model_type == 'arima':
                    print("Training ARIMA model...")
                    model = train_arima_model(stock_data)
                    predictions = predict_arima(model)

                elif model_type == 'prophet':
                    print("Training Prophet model...")
                    model = train_prophet_model(stock_data)
                    predictions = predict_prophet(model)

                elif model_type == 'random_forest':
                    print("Training Random Forest model...")
                    model = train_random_forest_model(stock_data)
                    predictions = predict_random_forest(model, stock_data)

                elif model_type == 'linear_regression':
                    print("Training Linear Regression model...")
                    model = train_linear_regression_model(stock_data)
                    predictions = predict_linear_regression(model, stock_data)

                elif model_type == 'moving_average':
                    print("Using Moving Average model...")
                    predictions = predict_moving_average(stock_data)

                else:
                    print("Error: Invalid model type.")
                    return go.Figure()

                # Create a date range for the predictions
                last_date = stock_data.index[-1]
                future_dates = pd.date_range(start=last_date, periods=31, freq='B')[1:]  # Exclude the last date

                # Ensure predictions and future_dates have the same length
                if len(predictions) != len(future_dates):
                    print("Error: Predictions and future_dates length mismatch.")
                    return go.Figure()

                # Plot the graph
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Data'))
                fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Predicted Data'))
                fig.update_layout(title=f"Stock Price Predictions for {stock_ticker}", xaxis_title="Date", yaxis_title="Price")
                return fig

            except Exception as e:
                print(f"Error in predictions: {e}")
                return go.Figure()

    return go.Figure()

@app.callback(
    Output('chat-response-1', 'children'),
    [Input('send-chat-1', 'n_clicks')],
    [State('chat-input-1', 'value'), State('stock-input-1', 'value')]
)
def update_chat_response_1(n_clicks, prompt, stock_ticker):
    if n_clicks > 0 and prompt and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            context = f"Stock Analysis for {stock_ticker}:\n{stock_data.tail().to_string()}"
            response = chat_with_cohere(prompt, context)
            return html.Div([
                html.H4('Cohere Response'),
                html.P(response)
            ])
    return "Ask me anything..."

@app.callback(
    Output('chat-response-2', 'children'),
    [Input('send-chat-2', 'n_clicks')],
    [State('chat-input-2', 'value'), State('stock-input-2', 'value')]
)
def update_chat_response_2(n_clicks, prompt, stock_ticker):
    if n_clicks > 0 and prompt and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            context = f"Monte Carlo Simulation for {stock_ticker}:\n{stock_data.tail().to_string()}"
            response = chat_with_cohere(prompt, context)
            return html.Div([
                html.H4('Cohere Response'),
                html.P(response)
            ])
    return "Ask me anything..."

@app.callback(
    Output('chat-response-3', 'children'),
    [Input('send-chat-3', 'n_clicks')],
    [State('chat-input-3', 'value'), State('stock-input-3', 'value')]
)
def update_chat_response_3(n_clicks, prompt, stock_ticker):
    if n_clicks > 0 and prompt and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            context = f"Financial Ratios for {stock_ticker}:\n{calculate_risk_metrics(stock_data)}"
            response = chat_with_cohere(prompt, context)
            return html.Div([
                html.H4('Cohere Response'),
                html.P(response)
            ])
    return "Ask me anything..."

@app.callback(
    Output('chat-response-4', 'children'),
    [Input('send-chat-4', 'n_clicks')],
    [State('chat-input-4', 'value'), State('stock-input-4', 'value')]
)
def update_chat_response_4(n_clicks, prompt, stock_ticker):
    if n_clicks > 0 and prompt and stock_ticker:
        articles = fetch_news(stock_ticker)
        if articles:
            context = f"News Sentiment for {stock_ticker}:\n{analyze_news_sentiment(articles)}"
            response = chat_with_cohere(prompt, context)
            return html.Div([
                html.H4('Cohere Response'),
                html.P(response)
            ])
    return "Ask me anything..."

@app.callback(
    Output('chat-response-5', 'children'),
    [Input('send-chat-5', 'n_clicks')],
    [State('chat-input-5', 'value'), State('stock-input-5', 'value')]
)
def update_chat_response_5(n_clicks, prompt, stock_ticker):
    if n_clicks > 0 and prompt and stock_ticker:
        articles = fetch_news(stock_ticker)
        if articles:
            context = f"Latest News for {stock_ticker}:\n{articles[:5]}"  # Display top 5 articles
            response = chat_with_cohere(prompt, context)
            return html.Div([
                html.H4('Cohere Response'),
                html.P(response)
            ])
    return "Ask me anything..."

@app.callback(
    Output('chat-response-6', 'children'),
    [Input('send-chat-6', 'n_clicks')],
    [State('chat-input-6', 'value'), State('stock-input-6', 'value'), State('period-input', 'value')]
)
def update_chat_response_6(n_clicks, prompt, stock_ticker, period):
    if n_clicks > 0 and prompt and stock_ticker and period:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            context = f"Recommendations for {stock_ticker}:\n{generate_recommendations(stock_data, calculate_risk_metrics(stock_data), period)}"
            response = chat_with_cohere(prompt, context)
            return html.Div([
                html.H4('Cohere Response'),
                html.P(response)
            ])
    return "Ask me anything..."

@app.callback(
    Output('chat-response-7', 'children'),
    [Input('send-chat-7', 'n_clicks')],
    [State('chat-input-7', 'value'), State('stock-input-7', 'value')]
)
def update_chat_response_7(n_clicks, prompt, stock_ticker):
    if n_clicks > 0 and prompt and stock_ticker:
        stock_data = fetch_stock_data(stock_ticker)
        if not stock_data.empty:
            context = f"Predictions for {stock_ticker}:\nPredictions will be displayed here."
            response = chat_with_cohere(prompt, context)
            return html.Div([
                html.H4('Cohere Response'),
                html.P(response)
            ])
    return "Ask me anything..."

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)
