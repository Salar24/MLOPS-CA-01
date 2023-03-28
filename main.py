from flask import Flask, render_template
import requests
import pandas as pd
from psx import stocks, tickers

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

tickers = tickers()
 
app = Flask(__name__)
            
@app.route("/predict")
def predict():
    model = LinearRegression()
    data = stocks("EFERT", start=datetime.date.today() - datetime.timedelta(days=365), end=datetime.date.today())
    print(data)
    data = data.reset_index()
    latest_data = data.iloc[-1].to_dict()
    print(latest_data)

    X = pd.DataFrame({'Open': data['Open'].values, 'High': data['High'].values, 'Low': data['Low'].values})
    y = data['Close'].values
    model.fit(X, y)

    predicted_dates = pd.date_range(start=datetime.date.today() - datetime.timedelta(days=365),
                                        end=datetime.date.today() + datetime.timedelta(days=1))
    future_data = pd.DataFrame({'Open': latest_data['Open'], 'High': latest_data['High'], 'Low': latest_data['Low']},
                                index=predicted_dates)
    future_close = model.predict(future_data)
    print(future_close[0])
    historical_data_x = data['Date']
    historical_data_y = data['Close']

    predicted_data_x = pd.date_range(start=datetime.date.today() - datetime.timedelta(days=365),
                                        end=datetime.date.today() + datetime.timedelta(days=1))
    newDictionary = {}
    newDictionary["Close"] = []
    print(newDictionary)
    let = 0
    for i in range(len(data)):
        predicted_dates = pd.date_range(start=datetime.date.today() - datetime.timedelta(days=365),
                                        end=datetime.date.today() - datetime.timedelta(days= 365-i))
        temporary_data = data.iloc[i].to_dict()
        filtered_data = pd.DataFrame({'Open': temporary_data['Open'], 'High': temporary_data['High'], 'Low': temporary_data['Low']},
                                index=predicted_dates)
        prediction = model.predict(filtered_data)
        newDictionary['Close'].append(prediction[0])
        let = let + 1

    predicted_data_y  = newDictionary['Close']

    print(len(data))
    print(f"Prediction for {latest_data['Date']} - Close: {future_close[0]:.2f}")
    for i in range(len(data)):
        print(predicted_data_y[i])
        print(predicted_data_x[i])
        print(historical_data_y[i])
        predicted_data_y[i] = round(predicted_data_y[i], 2)

    return render_template("form.html", pred = predicted_data_y, dates = predicted_data_x, actual = historical_data_y)






@app.route("/")
def user():
    fig = make_subplots(rows=1,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('5 Metrics'))

    historical_data = go.Scatter(x=[],
                             y=[],
                             mode='lines',
                             name='Historical Data',
                             line=dict(color='red', width=2))

    predicted_data = go.Scatter(x=[],
                            y=[],
                            mode='lines',
                            name='Predicted Data',
                            line=dict(color='green', width=2))

    fig.add_trace(historical_data)
    fig.add_trace(predicted_data)

    fig.update_layout(title="EFERT Stocks",
                    yaxis_title="Price (PKR)",
                    width=1400,
                    height=700)

    model = LinearRegression()

    while True:
        data = stocks("EFERT", start=datetime.date.today() - datetime.timedelta(days=365), end=datetime.date.today())
        print(data)
        data = data.reset_index()
        latest_data = data.iloc[-1].to_dict()
        print(latest_data)

        X = pd.DataFrame({'Open': data['Open'].values, 'High': data['High'].values, 'Low': data['Low'].values})
        y = data['Close'].values
        x_train, x_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

        model.fit(X, y)

        predicted_dates = pd.date_range(start=datetime.date.today() - datetime.timedelta(days=365),
                                        end=datetime.date.today() + datetime.timedelta(days=1))
        future_data = pd.DataFrame({'Open': latest_data['Open'], 'High': latest_data['High'], 'Low': latest_data['Low']},
                                index=predicted_dates)
        future_close = model.predict(future_data)
        print(future_close[0])
        historical_data.x = data['Date']
        historical_data.y = data['Close']
        
        predicted_data.x = data['Date']
        newDictionary = {}
        newDictionary["Close"] = []
        print(newDictionary)
        let = 0
        for i in range(len(data)):
            predicted_dates = pd.date_range(start=datetime.date.today() - datetime.timedelta(days=365),
                                        end=datetime.date.today() - datetime.timedelta(days= 365-i))
            temporary_data = data.iloc[i].to_dict()
            filtered_data = pd.DataFrame({'Open': temporary_data['Open'], 'High': temporary_data['High'], 'Low': temporary_data['Low']},
                                index=predicted_dates)
            prediction = model.predict(filtered_data)
            newDictionary['Close'].append(prediction[0])
            let = let + 1

        predicted_data.y = newDictionary['Close']
        print(len(data))
        print(f"Prediction for {latest_data['Date']} - Close: {future_close[0]:.2f}")

        fig.update_traces(historical_data, selector=dict(name='Historical Data'))
        fig.update_traces(predicted_data, selector=dict(name='Predicted Data'))

        fig.show()

        time.sleep(50)  # Wait for 5 minutes before updating the chart



if __name__ == "__main__":
    app.run()
