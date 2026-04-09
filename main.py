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
    X_raw = df.drop('species',axis = 1) #drop species เพราะ เป็น target 
    X_raw

    st.write('**y**')
    y_raw = df.species #เลือก column species มาขึ้นตาราง
    y_raw

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
    sex_v = st.selectbox('Sex',('Male','Female')) #ใช้ได้เหมือนกับ island แค่ไม่ได้กำหนด keyword แต่ต้องเรียง argument ที่ใส่
    
    # Create DataFrame for the in put Features
    #จริงๆไม่ต้อง TAB มาอยู่ sidebar ก็ได้เพราะมัน เป็น Global variable แค่ทำให้รู้ว่าทำได้
    data ={
        'island':island,
        'bill_length_mm' : bill_lenght,
        'bill_depth_mm' : bill_dept,
        'flipper_length_mm' : flipper_length,
        'body_mass_g' : body_mass,
        'sex' : sex_v
    }
    

inp_data = pd.DataFrame(data , index = [0])    
#inp_data #แสดงผลบน sidebar 💀
inp_penguins = pd.concat([inp_data , X_raw], axis=0)

with st.expander("Data and Data Combined"):
    st.write('**Interactive Data')
    inp_data
    st.write('**Combined Data X**')
    inp_penguins

#enconding X
encode = ['island','sex'] #กำหนด column ที่ต้อง encode object -> numeric
df_encoded = pd.get_dummies(inp_penguins,prefix=encode) #one hot encoding
#Tips ถ้าทำ ตัวแปรมีเยอะเสี่ยง multicolinearity ถ้าเรามีตัว 'cooked' , 'cooking' , 'dead' หาก 0,0 แปลว่า 'dead' = 1 ดังนั้นเราสามารถลบออกได้ 1 column เพื่อลดความซับซ้อนของข้อมูล (งงๆ แต่เก็บไว้เป็น concept) 
#df_penguins.head(3)#ใช้ได้แต่ไม่ขึ้น streamlit
#df_encoded[:5:] #work on streamlit

with st.expander('Encoded'):
    st.write('**Encoded Data Frame**')
    df_encoded[:5:] 
    
    
#Encoding y
target_mapper = {
    'Adelie' : 0,
    'Chinstrap' : 1,
    'Gentoo' : 2
}
def target_encode(val): #val = value
    return target_mapper[val] 
    
y = y_raw.apply(target_encode)
y