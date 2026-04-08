import streamlit as st
import pandas as pd
import os

st.title('🤖 Machine Learning App')

st.markdown('## Hi! This App is for study about Machine Learning and WEB APP for me!')

df = pd.read_csv("assets/penguins_cleaned.csv")
df