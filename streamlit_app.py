import streamlit as st
import pandas as pd
import os

st.title('🤖 Machine Learning App')

st.markdown('## Hi! This App is for study about Machine Learning and WEB APP for me!')

with st.expander('Data'):
    st.write('Raw Data')
    df = pd.read_csv("assets/penguins_cleaned.csv")
    df