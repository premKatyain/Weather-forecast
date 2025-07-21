from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY='7591cf60b81859be9cdea304509de2d2'
BASE_URL='https://api.openweathermap.org/data/2.5/'


def get_current_weather(city):
  url=f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
  response=requests.get(url)
  data=response.json()
  return {
      'city':data['name'],
      'current_temp':round(data['main']['temp']),
      'feels_like':round(data['main']['feels_like']),
      'temp_min':round(data['main']['temp_min']),
      'temp_max':round(data['main']['temp_max']),
      'humidity':round(data['main']['humidity']),
      'description':data['weather'][0]['description'],
      'country':data['sys']['country'],
      'wind_gust_dir':data['wind']['deg'],
      'wind_gust_speed':data['wind']['speed'],
      'pressure':data['main']['pressure'],
      'clouds':data['clouds']['all'],
      'visibility':data['visibility'],
  }
  

#reading historical data

def read_historical_data(filename):
  df=pd.read_csv(filename)
  df=df.dropna()
  df=df.drop_duplicates() # Corrected method
  return df


#prepare data for training

def prepare_data(data):
  le=LabelEncoder() 
  data['WindGustDir']=le.fit_transform(data['WindGustDir'])
  data['RainTomorrow']=le.fit_transform(data['RainTomorrow'])

  #define the feature variables and target variables
  X=data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']]
  y=data['RainTomorrow'] #target variable

  return X,y,le 


#Train a model

def train_rain_model(X,y):
   X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
   model=RandomForestClassifier(n_estimators=100,random_state=42)
   model.fit(X_train,y_train)

   y_pred=model.predict(X_test)

   print("Mean squared error for rain model")
   print(mean_squared_error(y_test,y_pred))
   
   return model


#prepare regression data

def prepare_regression_data(data,feature):
  X,y=[],[]

  for i in range(len(data)-1):
    X.append(data[feature].iloc[i])
    y.append(data[feature].iloc[i+1])
  X=np.array(X).reshape(-1,1)
  y=np.array(y)
  return X,y


#Train Regression Model

def train_regression_model(X,y):
  model=RandomForestRegressor(n_estimators=100,random_state=42)
  model.fit(X,y)
  return model


#Predict future

def predict_future(model,current_value):
  predictions=[current_value]
  for i in range(5):
    nex_value=model.predict(np.array([[predictions[-1]]]))
    predictions.append(nex_value[0])
  return predictions[1:]





def weather_view(request):
    if request.method=='POST':
        city=request.POST.get('city')
        current_weather=get_current_weather(city)
        csv_path=os.path.join('D:\\Data Analysis\\Machine Learning\\Weather\\weather_data.csv')

        historical_data=read_historical_data(csv_path)

        X,y,le=prepare_data(historical_data)
        rain_model=train_rain_model(X,y)

        X,y=prepare_regression_data(historical_data,'MinTemp')
        temp_model=train_regression_model(X,y)


        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]

        # Safely get compass direction with a fallback to 'N'
        compass_direction = next(
            (point for point, start, end in compass_points if start <= wind_deg < end),
            "N"
        )

        # Only transform if present in encoder classes
        if compass_direction in le.classes_:
            compass_direction_encoded = le.transform([compass_direction])[0]
        else:
            compass_direction_encoded = 0  # or handle/log as unknown



        current_data={
            'MinTemp':current_weather['temp_min'],
            'MaxTemp':current_weather['temp_max'],
            'WindGustDir':compass_direction_encoded,
            'WindGustSpeed':current_weather['wind_gust_speed'],
            'Humidity':current_weather['humidity'],
            'Pressure':current_weather['pressure'],
            'Temp':current_weather['current_temp']
        }

        current_df=pd.DataFrame([current_data])


        rain_prediction=rain_model.predict(current_df)[0]

        X_temp, y_temp=prepare_regression_data(historical_data,'Temp')

        X_hum, y_hum=prepare_regression_data(historical_data,'Humidity')

        temp_model=train_regression_model(X_temp,y_temp)
        hum_model=train_regression_model(X_hum,y_hum)

        future_temp=predict_future(temp_model,current_weather['temp_min'])
        future_hum=predict_future(hum_model,current_weather['humidity'])

        #prepare time for future predictions

        timezone=pytz.timezone('Asia/Kolkata')
        current_time=datetime.now(timezone)

        next_hour=current_time+timedelta(hours=1)
        next_hour=next_hour.replace(minute=0,second=0,microsecond=0)

        future_times=[(next_hour+timedelta(hours=i)).strftime('%H:00') for i in range(5)]


        time1,time2,time3,time4,time5=future_times
        temp1,temp2,temp3,temp4,temp5=future_temp
        hum1,hum2,hum3,hum4,hum5=future_hum
        
        #pass data to template
        
        context={
            'location':city,
            'current_temp':current_weather['current_temp'],
            'MinTemp':current_weather['temp_min'],
            'MaxTemp':current_weather['temp_max'],
            'feels_like':current_weather['feels_like'],
            'humidity':current_weather['humidity'],
            'clouds':current_weather['clouds'],
            'description':current_weather['description'],
            'city':current_weather['city'],
            'country':current_weather['country'],
            
            'time':datetime.now(),
            'date':datetime.now().strftime("%B %d, %Y"),
            
            'wind':current_weather['wind_gust_speed'],
            'pressure':current_weather['pressure'],
            'visibility':current_weather['visibility'],
            
            'time1':time1,
            'time2':time2,
            'time3':time3,
            'time4':time4,
            'time5':time5,
            
            'temp1':f"{round(temp1,1)}",
            'temp2':f"{round(temp2,1)}",
            'temp3':f"{round(temp3,1)}",
            'temp4':f"{round(temp4,1)}",
            'temp5':f"{round(temp5,1)}",
            
            'hum1':f"{round(hum1,1)}",
            'hum2':f"{round(hum2,1)}",
            'hum3':f"{round(hum3,1)}",
            'hum4':f"{round(hum4,1)}",
            'hum5':f"{round(hum5,1)}",
            
        }
        
        return render(request,'weather.html',context)
        
    return render(request,'weather.html')