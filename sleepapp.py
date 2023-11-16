import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Sleep disorder prediction")
#@st.cache(allow_output_mutation=True)
def get_model():
    return load_model("sleep_lifestyle")

def predict(model, data):
    prediction = predict_model(model, data=data)
    return prediction['prediction_label'][0]

model = get_model()
st.title("Sleep disorder prediction")

form = st.form('Sleep Disorder')

Gender = form.radio('Gender', ['male', 'female'])
Age = form.number_input('Age', min_value=1, max_value=100, value=25)
Occupation_list = ['Software Engineer','Doctor','Sales Representative','Teacher',
                   'Nurse','Engineer','Accountant','Scientist','Lawyer','Salesperson','Manager']
Occupation = form.selectbox('Occupation', Occupation_list)
Sleep_Duration = form.slider('Sleep Duration', min_value=1.0, max_value=10.0, value=1.0)
Quality_of_sleep = form.slider('Quality of Sleep', min_value=0, max_value=10, value=0)
Physical_Activity_level = form.slider('Physical Activity Level',min_value=0, max_value=100, value=30)
Stress_level = form.slider('Stress Level', min_value=0, max_value=10, value=3)
bmi_list = ['Normal','Overweight','Obese']
Bmi_category = form.selectbox('BMI Category',bmi_list)
heart_rate = form.slider('Heart Rate', min_value=50, max_value=100, value=72)
Daily_steps = form.slider('Daily Steps', min_value=1500, max_value=10000, value=3000)
Systolic = form.slider('Systolic', min_value=70, max_value=180, value=120)
Diastolic = form.slider('Diastolic', min_value=30, max_value=120, value=80)

predict_button = form.form_submit_button('Predict')

input_dict = {'Gender': Gender,
              'Age': Age,
              'Occupation':Occupation,
              'Sleep Duration':Sleep_Duration,
              'Quality of Sleep':Quality_of_sleep,
              'Physical Activity Level': Physical_Activity_level,
              'Stress Level':Stress_level,
              'BMI Category':Bmi_category,
              'Heart Rate':heart_rate,
              'Daily Steps':Daily_steps,
              'Systolic':Systolic,
              'Diastolic':Diastolic}

input_df = pd.DataFrame([input_dict])

if predict_button:
    output = predict(model, input_df)

    st.success("You have {}".format(output))



