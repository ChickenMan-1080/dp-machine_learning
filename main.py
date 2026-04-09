#เรียก module object
#เพิ่งรู้ library python มองเป็น object ที่มี class เป็น module
#ส่วน st แน่นอนคือแค่ชื่อเล่น(Alias)
#st.sidebar เข้าถึง Attribute -> Attribute คือ Object ประเภทนึงที่ pythod มอง

import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('Hi! This App is for study about Machine Learning and WEB APP for me!')

with st.expander('Data'):
    st.write('Raw Data')
    df = pd.read_csv("assets/penguins_cleaned.csv")
    df
    
    st.write('**X**') #
    X = df.drop('species',axis = 1) #drop species เพราะ เป็น target 
    X

    st.write('**y**')
    y = df.species #เลือก column species มาขึ้นตาราง
    y

with st.expander("Data Visualization"):
    st.scatter_chart(
        data=df,
        x='bill_length_mm',
        y='body_mass_g',
        color='species'
    )  
    
#Data Preparation
#context manager (with statement)
with st.sidebar: #ควบคุมตัวแปร x ทั้งหมด
    st.header("Input Feature")
    island = st.selectbox(label ='Island',
                          options=("Torgersen","Biscoe","Dream"))
    bill_lenght = st.slider('Bill Length(mm)',32.1,59.6,44.5)
    bill_dept = st.slider('Bill Dept(mm)',13.1,21.5,17.3)
    flipper_length = st.slider('Flipper Length(mm)',172,231,197)
    body_mass = st.slider('Body Mass(g)',2700,6300,4050)
    gender = st.selectbox('Gender',('Male','Female')) #ใช้ได้เหมือนกับ island แค่ไม่ได้กำหนด keyword แต่ต้องเรียง argument ที่ใส่
    
    # Create DataFrame for the in put Features
    #จริงๆไม่ต้อง TAB มาอยู่ sidebar ก็ได้เพราะมัน เป็น Global variable อยู่แค่ทำให้รู้ว่าทำได้
    data ={
        'island':island,
        'bill_lenght_(mm)' : bill_lenght,
        'bill_dept_(mm)' : bill_dept,
        'flipper_length_(mm)' : flipper_length,
        'body_mass_(g)' : body_mass,
        'gender' : gender
    }
    
    inp_data = pd.DataFrame(data , index = [0])
    #inp_data #แสดงผลบน sidebar 💀
inp_data