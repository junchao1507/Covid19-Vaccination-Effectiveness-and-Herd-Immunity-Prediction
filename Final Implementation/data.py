
"""
Created on Sun Aug  1 19:44:43 2021

@author: BernardBB
"""

import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd


def app():
    st.title('Data')
    st.write("This is the `Data` page of the proposal.")
    st.write("The following is the DataFrame of the COVID dataset.")
    st.write("\n\n")
    st.write("This is the `US Data` dataframe. You may filter the dataframe according to the columns.")
   
    df_us = obtainData()
    df_us = dropColumns(df_us)
    
    writeMarkdown()
    
    st.markdown('**Data Source:** [github.com](https://github.com/owid/covid-19-data/tree/master/public/data)')
    st.write('The size of the dataframe is ' + f"`{df_us.shape}`")
    createTable(df_us)
    
    
    
    
def obtainData():
    df_world = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Y2S3\Introduction to Data Science\0127122_0128664_0131545_group_assignment\Final Implementation')
    df_us = df_world[df_world['iso_code'] == 'USA']
    return df_us
    
   
    
def createTable(df_us):
    selected_columns = st.sidebar.multiselect('Columns', df_us.columns.values, df_us.columns.values)
    df_us_sorted = df_us[selected_columns]
    st.dataframe(df_us_sorted, width=1500, height=700)

    
def dropColumns(df_us):
    columnsToDrop = ['iso_code','continent','reproduction_rate', 'stringency_index', 'excess_mortality', 
                     'human_development_index', 'life_expectancy', 'hospital_beds_per_thousand', 'handwashing_facilities',
                     'female_smokers', 'male_smokers', 'diabetes_prevalence', 'cardiovasc_death_rate']
    df_us = obtainData()
    df_us = df_us.drop(columnsToDrop, 1, inplace=False)
    return df_us
    

def writeMarkdown():
    st.markdown("""
                This app performs a study of the impact of the COVID Pandemic.\n
                The research question of the study are:\n
                1. How does one dose of vaccination compare with two doses of vaccination in the effectiveness of reducing the number of positive Covid-19 cases and the number of deaths due to the Covid-19 in USA?\n
                """)
    st.markdown("""
                Independent:
                * people_fully_vaccinated
                * people_partially_vaccinated
                """)
    st.markdown("""
                Dependent:
                * new_cases_smoothed
                * new_deaths_smoothed
                """)
    st.markdown("""2. How many days does it take for the US to attain herd immunity (70% vaccinated)?
                """)
    st.markdown("""
                Independent:
                * percent_pop_vaccinated
                """)
    st.markdown("""
                Dependent:
                * date
                """)
    st.markdown("##")

