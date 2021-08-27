# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:44:43 2021

@author: BernardBB
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import plotly.graph_objects as go
import plotly.express as px

import eda

def app():
    st.title('Prediction')
    st.write("This is the `Prediction` page of the proposal.")
    st.write("The following is the prediction made from our model on the dataset.")
    
    #Droping all other columns except date and all_var
    varia = eda.sortVariableHeatMap()
    
    df_us = varia[0]
    rq1_vars = varia[1]
    rq1_ind = varia[2]
    rq1_dep = varia[3]
    rq2_vars = varia[4]
    rq2_ind = varia[5]
    rq2_dep = varia[6]
    keep_var = rq1_vars + rq2_vars
    
    df_us.drop(df_us.columns.difference(keep_var), 1, inplace=True)
    
    #Dropping unwanted rows
    idx = df_us.index[df_us['date'] == pd.to_datetime("2021-01-18", format = '%Y-%m-%d')].tolist()
    df_us.drop(df_us[
          (df_us.index <= idx[0]) |
          (df_us['date'] == pd.to_datetime("2021-02-15", format = '%Y-%m-%d')) |
          (df_us['date'] == pd.to_datetime("2021-05-31", format = '%Y-%m-%d')) |
          (df_us['date'] == pd.to_datetime("2021-06-18", format = '%Y-%m-%d')) |
          (df_us['date'] == pd.to_datetime("2021-07-05", format = '%Y-%m-%d')) |
          (df_us['date'] == pd.to_datetime("2021-07-31", format = '%Y-%m-%d')) ].index, inplace=True)

    st.dataframe(df_us.head())
    st.dataframe(df_us.tail())
    st.write(df_us.shape)
    
    
    ### Prediction model initialization
    rqVaria = declareNumpy(df_us)
    x1,x2,x3,y1,y2,y3 = rqVaria[0],rqVaria[1],rqVaria[2],rqVaria[3],rqVaria[4],rqVaria[5]
    
       ##Model 1
    m1 = Model(x1,y1, rq1_ind[0],rq1_dep[0], "Model 1: Full Vaccination vs New Cases")
    m1.build_poly_model()
    m1.plot_model()
    m1.compute_r2_score()


### Declaring function for independent and dependent variables for (RQ1 and RQ2)
def declareNumpy(dataframe):
    x1 = np.array(dataframe['people_fully_vaccinated'])
    x2 = np.array(dataframe['people_partially_vaccinated'])
    
    y1 = np.array(dataframe['new_cases_smoothed'])
    y2 = np.array(dataframe['new_deaths_smoothed'])
    
    x3 = np.array(dataframe['date'])
    y3 = np.array(dataframe['percent_pop_vaccinated'])
    
    return [x1,x2,x3,y1,y2,y3]
    


### A cubic function to be used in curve_fit to find the best curve for a dataset (no longer used)
def cubic_func(x,a,b,c,d):
    return a * x ** 3 + b * x ** 2 + c * x + d

### A quatic function to be used in curve_fit to find the best curve for a dataset
def quatic_func(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


#### A class that can build both polynomial and linear regression models 
class Model:
    def __init__(self, x, y, x_name, y_name, model_name):
        self.x = x
        self.y = y
        self.x_train = []
        self.x_train_date = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.y_train_pred = []
        self.y_test_pred = []
        self.x_name = x_name
        self.y_name = y_name
        self.const = []
        self.lm = object
        self.model_name = model_name

    ### A method to get the curve coefficients of a best fit curve
    def get_curve_coef(self):
        self.const = curve_fit(quatic_func, self.x_train, self.y_train)
        return self.const[0][0], self.const[0][1], self.const[0][2], self.const[0][3], self.const[0][4]
    
    ### A method to prepare training and testing data 
    def prepare_train_test_data(self):
        ## Performing train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=0.7, test_size=0.3, random_state=100)
        
        ## Converting x_train date datatype (if it is a datetype)
        if self.x_train.dtype == '<M8[ns]':
            self.x_train_date = ['{}-{}-{}'.format(y,m,d) for y, m, d in map(lambda x: str(x).split('-'), self.x_train)]# save this to plot regression graph
            self.x_train = pd.to_datetime(self.x_train)
            self.x_train = self.x_train.map(dt.datetime.toordinal)
            self.x_train = self.x_train.values
            self.x_test = pd.to_datetime(self.x_test)
            self.x_test = self.x_test.map(dt.datetime.toordinal)  
            self.x_test = self.x_test.values
        
    
    
    ### A method to build polynomial regression models (quatic to be specific)
    def build_poly_model(self):
        # Calling the prep method
        self.prepare_train_test_data()

        ## Building the model
        # Getting the best fit polynomial curve coeffificients
        self.get_curve_coef()
        # Fitting the coefficients into the quatic function and the assign to y train and test variables
        self.y_train_pred = quatic_func(self.x_train.reshape(-1, 1), self.const[0][0], self.const[0][1], self.const[0][2], self.const[0][3], self.const[0][4])
        st.write(self.x_train.reshape(-1,1))
        self.y_test_pred = quatic_func(self.x_test.reshape(-1, 1), self.const[0][0], self.const[0][1], self.const[0][2], self.const[0][3], self.const[0][4])
        
        
    ### A method to build linear regression models 
    def build_linear_model(self):
        self.prepare_train_test_data()
        self.lm = LinearRegression()
        self.lm.fit(self.x_train.reshape(-1, 1), self.y_train.reshape(-1, 1))
        self.y_train_pred = self.lm.predict(self.x_train.reshape(-1, 1))
        self.y_test_pred = self.lm.predict(self.x_test.reshape(-1, 1))
    
    
    ### A method to plot numerical models
    def plot_model(self):
        # Ensuring the values in the array are ordered
        orders = np.argsort(self.x_train.ravel())
        # Formatting the y ticks (the y value display on graphs)
        formatter = FuncFormatter(eda.format_yticks)
        
        """
        ## Plot the model
        fig, ax = plt.subplots()
        ax.scatter(self.x_train, self.y_train, color='blue', label='Actual')
        ax.plot(self.x_train[orders], self.y_train_pred[orders], color = 'red', label='Predicted')
        ax.tick_params(labelrotation=45)
        ax.set_title(self.model_name, fontsize = 15)
        ax.set_xlabel(self.x_name, fontsize = 15)
        ax.set_ylabel(self.y_name, fontsize = 15)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', labelsize = 12)
        ax.tick_params(axis='y', labelsize = 12)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        
        #plt.show()
        st.write(fig)
        """

        fig = go.Figure()
        fig.add_trace(go.Scatter(x = self.x_train, y = self.y_train, mode='markers', name='Actual'))
        
        fig.add_trace(go.Scatter(x = self.x_train[orders], y = self.y_train_pred[orders], mode='lines', name = 'Predicted'))
        fig.update_layout(xaxis_tickformat = '.2s',
                  yaxis_tickformat = '.2s',
                  width = 900,
                  height = 600,
                  font = dict(
                      color='#ffffff',
                      size=15),
                  paper_bgcolor='#403834',
                  plot_bgcolor='#ffffff')
        
        st.write(self.model_name)
        st.plotly_chart(fig)
        
    
    ### A method to plot models with datetime independent variable
    def plot_model_with_date(self):
        # Ensuring the values in the array are ordered
        orders = np.argsort(self.x_train.ravel()) 
        
        ## Plotting the model
        fig, ax = plt.subplots()
        ax.scatter(self.x_train, self.y_train, color='blue', label='Actual')
        ax.plot(self.x_train[orders], self.y_train_pred[orders], color = 'red', label='Predicted')
        
        self.x_train = self.x_train[orders]
        # Formatting the x_labels in such that oen date is displayed on the graph in every 10 dates
        x_labels = ['' if i%10 != 0 else dt.date.fromordinal(j).strftime('%Y-%m-%d') 
                    for i, j in enumerate(self.x_train)]
        ax.set_xticks(self.x_train)
        ax.set_xticklabels(x_labels)
        ax.tick_params(labelrotation=45)
        ax.set_title(self.model_name, fontsize = 15)
        ax.set_xlabel(self.x_name, fontsize = 15)
        ax.set_ylabel(self.y_name, fontsize = 15)
        ax.tick_params(axis='x', labelsize = 12)
        ax.tick_params(axis='y', labelsize = 12)
        fig.set_figheight(8)
        fig.set_figwidth(15)
        
        plt.show()
    
    
    ### A mthod to compute the r-squared score of models
    def compute_r2_score(self):
        print("----- R2 Score Report -----")
        print("Model: ", self.model_name)
        print("Train Set: ", r2_score(self.y_train, self.y_train_pred))
        print("Test Set: ", r2_score(self.y_test, self.y_test_pred))
        
        
    ### A method to predict future values of polynomial models
    def predict_poly(self, val):
        pred = []
        for i in val:
            pred.append(quatic_func(i, self.const[0][0], self.const[0][1], self.const[0][2], self.const[0][3], self.const[0][4]))
        return pred
    
    ### A method to test the polynomial models using user input
    def test_poly_model(self):
        value = input('Enter {0} in millions: '.format(self.x_name))
        value = int(value) * 1000000
        result = self.predict_poly([value])
        print('The predicted {0} is {1}'.format(self.y_name, result))
        
    ### A method to test the linear models using user input
    def test_linear_model(self):
        date = input('Enter {0} (yyyy-mm-dd): '.format(self.x_name))
        date = dt.datetime.strptime(date, "%Y-%m-%d")
        date = dt.date.toordinal(date)
        result = self.lm.predict([[date]])
        print('The predicted {0} is {1}'.format(self.y_name, result))