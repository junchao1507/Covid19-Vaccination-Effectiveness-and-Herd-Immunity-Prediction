# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:44:44 2021

@author: BernardBB
"""


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
#%matplotlib inline

import plotly.graph_objects as go


import data

def app():
    st.title('EDA')
    st.write("This is the `EDA` page of the proposal.")
    st.write("The following is the EDA Analysis conducted on the COVID dataset.")

    st.write("")
    varia = sortVariableHeatMap()
    df_us = varia[0]
    rq1_ind = varia[2]
    rq1_dep = varia[3]
    generate_heat_map(rq1_ind, df_us)
    st.markdown('The EDA shows a high correlation between a high correlation between people_fully_vaccinated and the dependent variables(new_cases_smoothed, new_deaths_smoothed).')
    st.markdown('In contrast, there is a lower correlation between people_partially_vaccinated and the dependent variables(new_cases_smoothed, new_deaths_smoothed).')
    
    
    st.write('#')
    st.title('Data Scatter plots')
    plot_scatter_plot(rq1_ind, rq1_dep, df_us)
    st.markdown('The shape of of the data scatter plots, which resembles cubic and quartic functions, induces the need for polynomic models to be used in the prediction models.')

# A function to plot a heat map of a list of independent variables against all other variables
def generate_heat_map(list_ind, df):
    corr = df.corr()
    fig = plt.figure(figsize = (20,25))
    plt.rcParams.update({'font.size': 22})
    sns.heatmap(corr[list_ind], cmap='coolwarm', annot = True)
    
    st.pyplot(fig)


# A function to format the yticks of graphs (display in millions)
def format_yticks(x, pos):
    return '%1.1fM' % (x * 1e-6)


# A function to plot scatter plot of a given list of independent and dependent variable
def plot_scatter_plot(list_ind, list_dep, df):
    ## Iterating through the dependent and independent variable list
    for i1, j1 in enumerate(list_dep):
        for i2, j2 in enumerate(list_ind):
            # Set title format
            title = "Ind: {} | Dep: {}".format(j2, j1)     
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[j2], y=df[j1], mode = 'markers', name="Ind: {} | Dep: {}".format(j2, j1)))
           
            fig.update_layout(xaxis_tickformat = '.2s',
                              yaxis_tickformat = '.2s',
                              xaxis_title = j2,
                              yaxis_title = j1,
                              width = 1100,
                              height = 700,
                              font = dict(
                                  color='#ffffff',
                                  size=15),
                              paper_bgcolor='#403834',
                              plot_bgcolor='#ffffff')
            
            st.write(title)
            st.plotly_chart(fig)
            st.markdown('#')
            


######### Functions for Data Cleaning
### A function to display a list of dates with a missing value for a list of specific variables.
def display_missing_val_dates(list_var, df):
    ## A variavle to count the total missing values
    missing = [0] * len(list_var)
    
    print("First index and date in the record: ", df['date'].iat[0],"\n")
    print("Last index and date in the record: ", df['date'].iat[0],"\n")
    
    ## Iterating through the given list of variable to display all missing data and count the sum of missing data
    for i, j in enumerate(list_var):
        print("Dates of Missing ", j, " data:")
        for index, row in df.iterrows():
            if pd.isnull(row[j]):
                print(row['date'])
                missing[i] += 1
        print("Total Missing Values: ", missing[i], "\n\n")  


### A function to group missing values of a list of specific variables by month.
def group_missing_val_by_month(list_var, dataframe):
    ## Create a month column
    date_str = []
    tmp_str = ''
    for index, row in dataframe.iterrows():
        tmp_str = str(row['date'])
        date_str.append(tmp_str[0:7])
    dataframe['month'] = date_str
    
    ## Printing Missing Values Groupby Month
    for i, j in enumerate(list_var):
        print('Group Missing values of', j, 'by month')
        missing = dataframe.groupby('month')
        print(missing.apply(lambda x: x[x[j].isna() == True]['month'].count()))
        print("\n")



def sortVariableHeatMap():
    ### Creating a new variable & define independent and dependant variable
    df_us = data.obtainData()
    
    # Creating a new variable: people_partially_vaccinated = people_vaccinated - people_fully_vaccinated
    df_us['people_partially_vaccinated'] = df_us['people_vaccinated'] - df_us['people_fully_vaccinated']
    
    # Creating a new variable: people_pop_vaccinated = people_fully_vaccinated / population * 100
    df_us['percent_pop_vaccinated'] = df_us['people_fully_vaccinated'] / df_us['population'] * 100
    
    #Reasigning the index of df_us
    df_us.index = df_us.index - 91593
    
    #Converting string date into datetime
    df_us['date'] = pd.to_datetime(df_us['date'], format = '%Y-%m-%d')

    ### Defining independent and dependent variables
    # Research Question 1
    rq1_ind = ['people_fully_vaccinated', 'people_partially_vaccinated']
    rq1_dep = ['new_cases_smoothed', 'new_deaths_smoothed']
    rq1_vars = rq1_ind + rq1_dep
    
    rq2_ind = ['date']
    rq2_dep = ['percent_pop_vaccinated']
    rq2_vars = rq2_ind + rq2_dep
    
    return [df_us, rq1_vars, rq1_ind, rq1_dep, rq2_vars, rq2_ind, rq2_dep]
    
    #st.write("Testing")
    #st.write(heatMap)