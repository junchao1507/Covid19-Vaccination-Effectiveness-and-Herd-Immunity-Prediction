# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 19:24:41 2021

@author: BernardBB
"""

import streamlit as st

class MultiApp:
    def __init__(self):
        self.apps = []
        
    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func})
        
    def run(self):
        app = st.sidebar.selectbox(
            'Navigation',
            self.apps,
            format_func=lambda app: app['title'])
        
        app['function']()
        