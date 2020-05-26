from flask import Flask,render_template,request,url_for
import pandas as pd 
import numpy as np 
import os
from models.model import arima_model
import glob
import chart_studio.plotly as py
from plotly.graph_objs import *
import pandas as pd
import chart_studio.plotly
import plotly.graph_objs as go
import plotly
import plotly.graph_objs as go

import json
import quandl
import pandas_datareader as pdr
import datetime
from dateutil import rrule, parser
from datetime import datetime, timedelta
import requests
import urllib.request
import time 
from bs4 import BeautifulSoup
from html.parser import HTMLParser
import lxml
# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

# NLP
# scientific computing library for saving, reading, and resizing images
import re
from textblob import TextBlob

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from keras.layers import Dropout

from statsmodels.tsa.arima_model import ARIMA
from math import sqrt

import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import model_from_json
import tensorflow as tf



def LSTM_Preprocess(data,choice):
        test=[]
        if choice==0:
                test=data.iloc[::-1, 1:2].values
        if choice==1:
                test=data.iloc[::-1, 2:3].values
        if choice==2:
                test=data.iloc[::-1, 3:4].values
        if choice==3:
                test=data.iloc[::-1, 7:8].values
        sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = sc.fit_transform(test)
        X_train = []
        for i in range(200, len(data)):
                X_train.append(training_set_scaled[i-200:i, 0])
        X_train = np.array(X_train)
        print(len(training_set_scaled))
        print(X_train.shape)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        return X_train,sc
        

def LSTM_model_load(model_file,choice):
        new_pkl_files=os.listdir()
        if choice==0:
                open_price_j=list(filter(lambda x: x[-6:] == 'o.json', new_pkl_files))
                open_price_h=list(filter(lambda x: x[-4:] == 'o.h5', new_pkl_files))
        if choice==1:
                open_price_j=list(filter(lambda x: x[-6:] == 'h.json', new_pkl_files))
                open_price_h=list(filter(lambda x: x[-4:] == 'h.h5', new_pkl_files))
        if choice==2:
                open_price_j=list(filter(lambda x: x[-6:] == 'l.json', new_pkl_files))
                open_price_h=list(filter(lambda x: x[-4:] == 'l.h5', new_pkl_files))
        if choice==3:
                open_price_j=list(filter(lambda x: x[-6:] == 't.json', new_pkl_files))
                open_price_h=list(filter(lambda x: x[-4:] == 't.h5', new_pkl_files))
        print(open_price_j)
        print(open_price_h)
        json_file = open(str(open_price_j[0]),'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        #load weights into new model
        loaded_model.load_weights(str(open_price_h[0]))
        print("Loaded Model from disk")
        #compile and evaluate loaded model
        loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        graph = tf.get_default_graph()
        print(str(choice)*19)
        return loaded_model,graph


def LSTM_Prediction(model,graph,x,sc):
        with graph.as_default():
                # perform the prediction
                out = model.predict(x)
                print(out)
                # convert the response to a string
                out1=sc.inverse_transform(out)
                return float(out1[-2])

__autor__="Jeeva"
APP_ROOT=os.path.dirname(os.path.abspath(__file__)) 

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')
@app.route('/upload',methods=['POST'])
def upload():
    target=os.path.join(APP_ROOT,'files/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist('file'):
        print(file)
        filename=file.filename
        destination ='/'.join([target,filename])
        print(destination)
        file.save(destination)
@app.route('/predict',methods=["GET","POST"])



        

def predict():
        if request.method == 'POST':
                raw_text = request.form['rawtext']
                print(raw_text)
                os.chdir(os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/models")
                models=os.listdir()
                model_file=list(filter(lambda x: x[:len(raw_text)] == raw_text, models))
                vl=0
                new_pkl_files=os.listdir()
                print(raw_text)
                print(model_file)
                if len(model_file)==0:
                        try:
                                print(raw_text)
                                date=datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
                                date1 = "2010-10-01"
                                date2 = date
                                datesx = list(rrule.rrule(rrule.DAILY, dtstart=parser.parse(date1), until=parser.parse(date2)))
                                try:
                                        aapl = pdr.get_data_yahoo(str(raw_text), start="2010-10-01", end=date)
                                        data=aapl.rename(columns={"Adj Close": "No of Trades","High": "High Price","Low": "Low Price","Open": "Open Price","Close": "Close Price","Volume":"No.of Shares"})
                                        print(data)
                                except:
                                        aapl = quandl.get("WIKI/"+str(raw_text), start_date="2010-10-01", end_date=str(date))
                                        data=aapl.rename(columns={"Adj. Close": "No of Trades","High": "High Price","Low": "Low Price","Open": "Open Price","Close": "Close Price","Volume":"No.of Shares"})
                                os.chdir(os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files")
                                print(data)
                                print(len(data),len(datesx))
                                datesx=datesx[-len(data):]
                                data["Date"]=datesx
                                print(len(data),len(datesx))
                                data.to_csv(str(raw_text)+ ".csv")
                                print(raw_text)
                                data.to_csv(str(raw_text)+ ".csv",index=False)
                                open_pred=arima_model(raw_text,0,os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files")
                                high_pred=arima_model(raw_text,1,os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files")
                                low_pred=arima_model(raw_text,2,os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files")
                                trad_pred=arima_model(raw_text,3,os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files")
                        except:
                                print('Stock is not Available will available in feature')
                        
                else:
                        data_df=pd.read_csv(os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files/" +str(raw_text)+ ".csv")
                        open_p,grap_o=LSTM_model_load(raw_text,0)
                        high_p,grap_h=LSTM_model_load(raw_text,1)
                        low_p,grap_l=LSTM_model_load(raw_text,2)
                        print("Hello")
                        data_df=pd.read_csv(os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files/" +str(raw_text)+ ".csv")
                        print(data_df.head())
                        print(data_df.tail())
                        test_o,open_sc=LSTM_Preprocess(data_df,0)
                        test_h,high_sc=LSTM_Preprocess(data_df,1)
                        test_l,low_sc=LSTM_Preprocess(data_df,2)
                        print("Hello")
                        
                        open_pred=LSTM_Prediction(open_p,grap_o,test_o,open_sc)
                        high_pred=LSTM_Prediction(high_p,grap_h,test_h,open_sc)
                        low_pred=LSTM_Prediction(low_p,grap_l,test_l,open_sc)
                        print(open_pred)
                        print(high_pred)
                        print(low_pred)
                        print(raw_text)
                        trad_pred=arima_model(raw_text,3,os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files")
                        print(trad_pred)
        print(raw_text)
        data_df=pd.read_csv(os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files/" +str(raw_text)+ ".csv")
        data = [go.Scatter(x=data_df["Date"][::-1],  y=data_df["No of Trades"][::-1],name="No of Traders" )]
        bar = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        data = [go.Scatter(x=data_df["Date"][::-1],  y=data_df["Open Price"][::-1],name="Open Price" )]
        bar1 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        data = [go.Scatter(x=data_df["Date"][::-1],  y=data_df["High Price"][::-1],name="High Price" )]
        bar2 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        data = [go.Scatter(x=data_df["Date"][::-1],  y=data_df["Low Price"][::-1],name="Low Price" )]
        bar3 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=data_df["Date"][::-1], y=data_df["No of Trades"],
                            mode='markers',
                            name='No of Traders'))
        fig.add_trace(go.Scatter(x=data_df["Date"][::-1], y=data_df["High Price"][::-1],
                            mode='lines+markers',
                            name='High Price'))
        fig.add_trace(go.Scatter(x=data_df["Date"][::-1], y=data_df["Low Price"][::-1],
                            mode='lines',
                            name='Low'))
        bar4 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        fig = go.Figure(data=[go.Candlestick(x=data_df["Date"][::-1],
                open=data_df["Open Price"][::-1],
                high=data_df["High Price"][::-1],
                low=data_df["Low Price"][::-1],
                close=data_df["Close Price"][::-1])])
        fig.update_layout(xaxis_rangeslider_visible=True)
        mid=[]
        for i in range(len(data_df)):
                pred_mid=(float(data_df["High Price"][i])+float(data_df["Low Price"][i]))/2.0
                mid.append(pred_mid)
        data = [go.Scatter(x=data_df["Date"][::-1],  y=mid[::-1],name="Average Price" )]
        bar7 = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
        bar5 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        fig = go.Figure()

        # Add surface trace
        fig.add_trace(go.Surface(z=data_df.values.tolist(), colorscale="Viridis"))

        # Update plot sizing
        fig.update_layout(
            width=1326,
            height=500,
            autosize=False,
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_white"
        )

        # Update 3D scene options
        fig.update_scenes(
            aspectratio=dict(x=1, y=1, z=0.7),
            aspectmode="manual"
        )

        # Add dropdown
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=["type", "surface"],
                            label="3D Surface",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "heatmap"],
                            label="Heatmap",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )

        # Add annotation
        fig.update_layout(
            annotations=[
                dict(text="Trace type:", showarrow=False,
                x=0, y=1.085, yref="paper", align="left")
            ]
        )
        bar6 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        df=data_df
        trace1 = {
          "name": "GS", 
          "type": "candlestick", 
          "x":df["Date"][::-1],
          "yaxis": "y2", 
          "low":df["Low Price"][::-1],
          "high": df["High Price"][::-1],
          "open": df["Open Price"][::-1],
          "close": df["Close Price"][::-1],
          "decreasing": {"line": {"color": "#7F7F7F"}}, 
          "increasing": {"line": {"color": "#17BECF"}}
        }
        mid=[]
        for i in range(len(df)):
            pred_mid=(float(df["High Price"][i])+float(df["Low Price"][i]))/2.0
            mid.append(pred_mid)
        trace2 = {
          "line": {"width": 1}, 
          "mode": "lines", 
          "name": "Moving Average", 
          "type": "scatter", 
          "x": df["Date"][::-1],
          "y":mid[::-1],
          "yaxis": "y2", 
          "marker": {"color": "#E377C2"}
        }
        trace3 = {
          "name": "Volume", 
          "type": "bar", 
          "x": df["Date"][::-1],
          "y": df["No.of Shares"][::-1],
          "yaxis": "y", 
          "marker": {"color": ["#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF", "#17BECF", "#7F7F7F", "#7F7F7F", "#17BECF", "#7F7F7F", "#17BECF", "#17BECF", "#17BECF"]}
        }
        high=[]
        low=[]
        for i in range(20):
            high.append(None)
            low.append(None)
        import statistics
        high2=df["High Price"][::-1]
        low2=df["Low Price"][::-1]

        for i in range(20,len(df)):
            high1=high2[i-20:i]
            low1=low2[i-20:i]
            highavg=sum(high1)/20.0
            lowavg=sum(low1)/20.0
            highstd=statistics.stdev(high1)*2
            lowstd=statistics.stdev(low1)*2
            high_vl=highavg+highstd
            low_vl=lowavg-lowstd
            high.append(high_vl)
            low.append(low_vl)

            
        trace4 = {
          "line": {"width": 1}, 
          "name": "Bollinger Bands", 
          "type": "scatter", 
          "x": df["Date"][::-1],
          "y": high,
          "yaxis": "y2", 
          "marker": {"color": "#ccc"}, 
          "hoverinfo": "none", 
          "legendgroup": "Bollinger Bands"
        }
        trace5 = {
          "line": {"width": 1}, 
          "type": "scatter", 
          "x": df["Date"][::-1],
          "y": low,
          "yaxis": "y2", 
          "marker": {"color": "#ccc"}, 
          "hoverinfo": "none", 
          "showlegend": False, 
          "legendgroup": "Bollinger Bands"
        }
        data = Data([trace1, trace2, trace3, trace4, trace5])
        layout = {
          "xaxis": {"rangeselector": {
              "x": 0, 
              "y": 0.9, 
              "font": {"size": 13},  
              "bgcolor": "rgba(150, 200, 250, 0.4)", 
              "buttons": [
                {
                  "step": "all", 
                  "count": 1, 
                  "label": "reset"
                }, 
                {
                  "step": "year", 
                  "count": 1, 
                  "label": "1yr", 
                  "stepmode": "backward"
                }, 
                {
                  "step": "month", 
                  "count": 3, 
                  "label": "3 mo", 
                  "stepmode": "backward"
                }, 
                {
                  "step": "month", 
                  "count": 1, 
                  "label": "1 mo", 
                  "stepmode": "backward"
                }, 
                {"step": "all"}
              ]
            }}, 
          "yaxis": {
            "domain": [0, 0.2], 
            "showticklabels": False
          }, 
          "legend": {
            "x": 0.3, 
            "y": 0.9, 
            "yanchor": "bottom", 
            "orientation": "h"
          }, 
          "margin": {
            "b": 40, 
            "l": 40, 
            "r": 40, 
            "t": 40
          }, 
          "yaxis2": {"domain": [0.2, 0.8]}, 
          "plot_bgcolor": "rgb(250, 250, 250)"
        }

        fig = Figure(data=data, layout=layout)
        bar8 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        trad_last=data_df["No of Trades"][::-1].iloc[-1]
        print(trad_last)
        open_last=data_df["Open Price"][::-1].iloc[-1]
        high_last=data_df["High Price"][::-1].iloc[-1]
        low_last=data_df["Low Price"][::-1].iloc[-1]
        last_mid=(float(high_pred)+float(low_last))/2.0
        print(last_mid)
        pred_mid=(float(high_last)+float(low_pred))/2.0
        Messge='MAKE YOUR OWN DECISION'
        if trad_pred<trad_last and pred_mid<last_mid:
                Messge='SELL THE STOCK'
        elif trad_pred<trad_last and pred_mid>last_mid:
                Messge='BUY THE STOCK'
        elif trad_pred>trad_last and pred_mid>last_mid:
                Messge='SELL THE STOCK'
        elif trad_pred>trad_last and pred_mid<last_mid:
                Messge='SELL THE STOCK'
        trad_pred=float(trad_pred)
        trad_pred=round(trad_pred)
        #trad_pred=trad_pred[1:-1]
        open_pred=float(open_pred)
        high_pred=float(high_pred)
        low_pred=float(low_pred)
        open_pred=round(open_pred , 2)
        high_pred=round(high_pred , 2)
        low_pred=round(low_pred , 2)
        
        print(str(trad_last),str(open_last),str(high_last),str(low_last),str(trad_pred),str(open_pred),str(high_pred),str(low_pred))
        os.chdir(os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+"/files")
        os.remove(str(raw_text)+ ".csv")
        return render_template('index.html',plot=bar,plot7=bar7,plot8=bar8,plot6=bar6,plot4=bar4,plot5=bar5,plot1=bar1,plot2=bar2,plot3=bar3,Message=Messge,nt1=str(trad_last),op1=str(open_last),hp1=str(high_last),lp1=str(low_last),nt2=str(trad_pred),op2=str(open_pred),hp2=str(high_pred),lp2=str(low_pred))


if __name__ == '__main__':
        app.run(debug=True)

		
