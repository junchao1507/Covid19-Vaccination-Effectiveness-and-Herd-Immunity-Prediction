
"""
Created on Sun Aug  1 19:44:43 2021

@author: BernardBB
"""

import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import plotly.graph_objects as go

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
    
    ### Prediction model initialization
    rqVaria = declareNumpy(df_us)
    x1,x2,x3,y1,y2,y3 = rqVaria[0],rqVaria[1],rqVaria[2],rqVaria[3],rqVaria[4],rqVaria[5]
    
       ##Model 1
    m1 = Model(x1,y1, rq1_ind[0],rq1_dep[0], "Model 1: Full Vaccination vs New Cases")
    m1.build_poly_model()
    m1.plot_model()
    m1.compute_r2_score()
    st.write('The regression plot indicates that there is a high accuracy value between the prediction model and the data.')
    st.write('This is backed up by the high r2 score for both the train set and test set.')
    st.markdown('#')
    st.write('You may input a specific value here to gain a prediction output.')
    m1.test_poly_model()
    st.markdown('##')
    st.markdown("#")
    
    m2 = Model(x1,y2, rq1_ind[0],rq1_dep[1], "Model 2: Full Vaccination vs New Deaths")
    m2.build_poly_model()
    m2.plot_model()
    m2.compute_r2_score()
    st.write('The regression plot indicates that there is a high accuracy value between the prediction model and the data.')
    st.write('This is backed up by the high r2 score for both the train set and test set.')
    st.markdown('#')
    st.write('You may input a specific value here to gain a prediction output.')
    m2.test_poly_model()
    st.markdown('##')
    st.markdown("#")

    m3 = Model(x2, y1, rq1_ind[1], rq1_dep[0], "Model 3: Partially Vaccination vs New Cases")
    m3.build_poly_model()
    m3.plot_model()
    m3.compute_r2_score()
    st.write('The regression plot indicates that there is a lower accuracy value between the prediction model and the data.')
    st.write('This is backed up by the lower r2 score for both the train set and test set in compared with the previous models.')
    st.write('This indicates that the people_partially_smoothed may not be very effective in curbing the cases of the COVID-19.')
    st.markdown('#')
    st.write('You may input a specific value here to gain a prediction output.')
    m3.test_poly_model()
    st.markdown('##')
    st.markdown("#")
    
    m4 = Model(x2, y2, rq1_ind[1], rq1_dep[1], "Model 4: Partially Vaccination vs New Deaths")
    m4.build_poly_model()
    m4.plot_model()
    m4.compute_r2_score()
    st.write('The regression plot indicates that there is a lower accuracy value between the prediction model and the data.')
    st.write('This is backed up by the lower r2 score for both the train set and test set in compared with the previous models.')
    st.write('This indicates that the people_partially_smoothed may not be very effective in curbing the deaths of the COVID-19.')
    st.markdown('#')
    st.write('You may input a specific value here to gain a prediction output.')
    m4.test_poly_model()
    st.markdown('##')
    st.markdown("#")


    m5 = Model(x3, y3, rq2_ind[0], rq2_dep[0], "Model 5: Date vs % of People Fully Vaccinated")
    m5.build_linear_model()
    m5.plot_model_with_date(df_us)
    m5.compute_r2_score()
    st.write('The regression plot indicates an upward trend during the duration of the pandemic.')
    st.write('The r2 score report also shows a high accuracy value, indicating the effectiveness of the model.')
    st.write('This validates the validity of our 2nd research question, in that a herd immunity will inevitably be achieved.')
    st.markdown('#')
    st.write('You may input a specific value here to gain a prediction output.')
    m5.test_linear_model()
    m5.test_linear_model_basedondate()

    st.markdown('#')
    
    
    st.markdown('##')
    st.markdown('##')
    
    plotComparisonModel(m1, m3)
    st.write('The effectiveness of people_fully_vaccinated and people_partially_vaccinated towards new_cases_smoothed is shown in the graph above.')
    st.write('As can be seen, the people_partially_vaccinated is not very effective in influencing the Covid-19 cases.')
    
    st.markdown('##')
    st.markdown('##')
    
    
    plotComparisonModel(m2, m4)
    st.write('The effectiveness of people_fully_vaccinated and people_partially_vaccinated towards new_deaths_smoothed is shown in the graph above.')
    st.write('As can be seen, the people_partially_vaccinated is not very effective in influencing the Covid-19 deaths.')
    
    st.markdown('##')
    st.markdown('##')
    
    
    

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

### A linear function to be used in linear_fit


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
        self.short_name = model_name[0:7]

    ### A method to get the curve coefficients of a best fit curve
    def get_curve_coef(self):
        self.const, ccov = curve_fit(quatic_func, self.x_train, self.y_train)
        #return self.const[0][0], self.const[0][1], self.const[0][2], self.const[0][3], self.const[0][4]
    
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
        self.y_train_pred = quatic_func(self.x_train, *self.const)
        self.y_test_pred = quatic_func(self.x_test, *self.const)
        
        
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
        
        #Creating the base figure to insert our graphs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = self.x_train, y = self.y_train, mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x = self.x_train[orders], y = self.y_train_pred[orders], mode='lines', name = 'Predicted'))
        
        #Figure design formatting
        fig.update_layout(xaxis_tickformat = '.2s',
                  yaxis_tickformat = '.2s',
                  xaxis_title = self.x_name,
                  yaxis_title = self.y_name,
                  width = 1100,
                  height = 700,
                  font = dict(
                      color='#ffffff',
                      size=15),
                    paper_bgcolor='#403834',
                    plot_bgcolor='#ffffff')
        
        st.write(self.model_name)
        st.plotly_chart(fig)
        
    
    ### A method to plot models with datetime independent variable
    def plot_model_with_date(self,df_us):
        # Ensuring the values in the array are ordered
        orders = np.argsort(self.x_train.ravel()) 
        
        self.y_train = self.y_train / 100
        self.y_train_pred = self.y_train_pred / 100
        
        
        subArray = []
        for i in self.y_train_pred[orders]:
            for j in i:
                subArray.append(j)
        self.y_train_pred = subArray
        
        
        
        #Creating the base figure to insert our graphs
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = self.x_train, y=self.y_train, mode='markers', name="Actual"))
        fig.add_trace(go.Scatter(x = self.x_train[orders], y = self.y_train_pred, mode='lines', name='Predicted'))
        
        self.x_train = self.x_train[orders]
        min_date = dt.datetime(2021,1,17)
        max_date = dt.datetime(2021,8,3)
        
        #Custom date_input in sidebar for model 5
        st.sidebar.markdown('##')
        st.sidebar.title('Model 5')
        min_date = st.sidebar.date_input("Select the Min Date", value=min_date ,min_value=min_date, max_value=max_date)
        max_date = st.sidebar.date_input("Select the Max Date", value=max_date ,min_value=min_date, max_value=max_date)
        
        min_date = dt.date.toordinal(min_date)
        max_date = dt.date.toordinal(max_date)
        
        #Figure design formatting
        fig.update_layout(
                  width = 1100,
                  height = 700,
                  yaxis_tickformat = '.00%',
                  xaxis_title = self.x_name,
                  yaxis_title = self.y_name,
                  xaxis_tickangle = 45,
                  font = dict(
                      color='#ffffff',
                      size=15),
                      margin=dict(
                        l=10,
                        r=10,
                        b=10,
                        t=10,
                        pad=2
                    ),
                    paper_bgcolor='#403834',
                    plot_bgcolor='#ffffff',
                  xaxis=dict(tickvals = self.x_train
                             #,ticktext = [dt.date.fromordinal(j).strftime('%Y-%m-%d')
                                          #for i, j in enumerate(self.x_train)]
                            ,ticktext = [' ' if i%5 != 0 else dt.date.fromordinal(j).strftime('%Y-%m-%d') 
                                      for i, j in enumerate(self.x_train)]
                            ,range = [min_date, max_date]
                  ))
        
        st.write(self.model_name)
        st.plotly_chart(fig)
        
        self.build_linear_model()
          
    
    ### A mthod to compute the r-squared score of models
    def compute_r2_score(self):
        st.write("----- R2 Score Report -----")
        st.write("Model: ", self.model_name)
        st.write("Train Set: ", r2_score(self.y_train, self.y_train_pred))
        st.write("Test Set: ", r2_score(self.y_test, self.y_test_pred))
        
    ### A method to predict future values of polynomial models
    def predict_poly(self, val):
        pred = []
        for i in val:
            pred.append(quatic_func(i, self.const[0], self.const[1], self.const[2], self.const[3], self.const[4]))
        return pred
    
    ### A method to test the polynomial models using user input
    def test_poly_model(self):
        value = st.number_input('Enter {0} in millions: '.format(self.x_name), key=self.model_name)
        st.write('The predicted {0} is {1}'.format(self.y_name, self.predict_poly([value * 1000000])))
        
        
        
    ### A method to test the linear models using user input
    def test_linear_model(self):
        min_date = dt.datetime(2021,1,1)
        max_date = dt.datetime(2023,1,1)
        
        date = st.date_input("Pick a date", min_value=min_date, max_value=max_date, value = min_date)
        date = dt.date.toordinal(date)
        st.write('The predicted {0} is {1}'.format(self.y_name, self.lm.predict([[date]])))
        
    
    def test_linear_model_basedondate(self):
        value = st.number_input('Enter {0} '.format(self.y_name))
        st.write('The predicted {0} is {1}'.format(self.x_name,date.fromordinal(int((np.roots((self.lm.coef_.item(), self.lm.intercept_ - value))[0])))))
        #st.write('The predicted {0} is {1}'.format(self.x_name,np.roots((self.lm.coef_.item(), self.lm.intercept_ - value))))
        
###Converting Ordinal Date to date



def plotComparisonModel(model1, model2):
    orders1 = np.argsort(model1.x_train.ravel())
    orders2 = np.argsort(model2.x_train.ravel())
    model1.build_poly_model
    model2.build_poly_model
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = model1.x_train[orders1], y = model1.y_train_pred[orders1], mode='lines', name = model1.short_name))
    fig.add_trace(go.Scatter(x = model2.x_train[orders2], y = model2.y_train_pred[orders2], mode='lines', name = model2.short_name))
    fig.update_layout(xaxis_tickformat = '.2s',
          yaxis_tickformat = '.2s',
          yaxis_title = model1.y_name,
          width = 1100,
          height = 700,
          font = dict(
              color='#ffffff',
              size=15),
          paper_bgcolor='#403834',
          plot_bgcolor='#ffffff')
    st.write('Comparison between {0} and {1}'.format(model1.model_name, model2.model_name))
    st.plotly_chart(fig)