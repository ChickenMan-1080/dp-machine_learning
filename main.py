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
    y = df.species #เลือก column species มาขึ้นตาราง
    y

with st.expander("Data Visualization"):
    st.scatter_chart(
        data=df,
        x='bill_length_mm',
        y='body_mass_g',
        color='species'
    )  
   
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
        'island':island, #island contain value that user select
        'bill_length_mm' : bill_lenght, #same as above
        'bill_depth_mm' : bill_dept, #same as above
        'flipper_length_mm' : flipper_length, #same as above
        'body_mass_g' : body_mass, #same as above
        'sex' : sex_v #same as above
    } #contain value in each column in dic data 
    

inp_data = pd.DataFrame(data , index = [0])    # create a dataframe for input data
#inp_data #แสดงผลบน sidebar 💀
inp_data_and_Xraw = pd.concat([inp_data , X_raw], axis=0) #combined inp_data and X_raw

with st.expander("Data and Data Combined"):
    st.write('**Interactive Data')
    inp_data
    st.write('**Combined Data X**')
    inp_data_and_Xraw


#Data Preparation

#enconding X
encode = ['island','sex'] #กำหนด column ที่ต้อง encode object -> numeric
df_encoded_X = pd.get_dummies(inp_data_and_Xraw,prefix=encode) #one hot encoding
#Tips ถ้าทำ ตัวแปรมีเยอะเสี่ยง multicolinearity ถ้าเรามีตัว 'cooked' , 'cooking' , 'dead' หาก 0,0 แปลว่า 'dead' = 1 ดังนั้นเราสามารถลบออกได้ 1 column เพื่อลดความซับซ้อนของข้อมูล (งงๆ แต่เก็บไว้เป็น concept) 
#df_penguins.head(3)#ใช้ได้แต่ไม่ขึ้น streamlit
#df_encoded[:5:] #work on streamlit


X_D = df_encoded_X[1::] 
    
#Encoding y
target_mapper = {
    'Adelie' : 0,
    'Chinstrap' : 1,
    'Gentoo' : 2
}
def target_encode(val): #val = value
    return target_mapper[val] 
    
encoded_y = y.apply(target_encode)

with st.expander('Data Preparation'): #inp_data_for_prediction
    st.write('**Encoded X (input data)')
    df_encoded_X
    st.write('**Encoded y')
    encoded_y
    
#Model Training and inference
#using random forest algorithm for train model

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()        #clf = classification
clf.fit(X_D,y)

# Apply model to make prediction
prediction = clf.predict(df_encoded_X[:1:]) #From line 78 maybe this reason why but he use variable but f it    i'll just select column instend
prediction_probs = clf.predict_proba(df_encoded_X[:1:]) #look prob all y

df_prediction_probs = pd.DataFrame(prediction_probs)


#df_prediction_probs.columns = ['Adelie','Chinstrap','Gentoo']
#df_prediction_probs.rename(
#    columns={
#        0:'Adelie',
#        1:'Chinstrap',
#        2:'Gentoo'
#    }
#) 

# This also work to change columns name but it's confused me so fxing much
# So i'll do what i understand like below


df_prediction_probs.rename(
    columns={
        0:'Adelie',
        1:'Chinstrap',
        2:'Gentoo'
    },
    inplace=True
)


#Display predicted species

import numpy as np

st.subheader('Predicted species')

st.dataframe(
    df_prediction_probs,
    column_config={ #column_config here is a keyword 
        'Adelie': st.column_config.ProgressColumn( #column_config here is a attribute access 
            'Adelie',
            format='%f',
            min_value=0,
            max_value=1
        ),
        'Chinstrap': st.column_config.ProgressColumn(
            'Chinstrap',
            format='%f',
            min_value=0,
            max_value=1
        ),
        'Gentoo': st.column_config.ProgressColumn(
            'Gentoo',
            format='%f',
            min_value=0,
            max_value=1
        )   
    },
    hide_index=True # hide first index i mean the annoying one 
)



penguin_species = np.array(['Adelie','Chinstrap','Gentoo']) #สร้าง numpy array แปลง list ให้เป็น numpy array
#st.success(str(penguin_species[prediction][0])) #error string array
st.success(prediction[0])
