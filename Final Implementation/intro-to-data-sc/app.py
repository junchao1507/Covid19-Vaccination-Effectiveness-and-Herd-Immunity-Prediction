# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:23:42 2021

@author: BernardBB
"""

from multiapp import MultiApp
import data, eda, prediction2

app = MultiApp()

app.add_app('Data', data.app)
app.add_app('EDA', eda.app)
app.add_app('Prediction', prediction2.app)

app.run()