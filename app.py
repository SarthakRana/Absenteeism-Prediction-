from absenteeism_module import *
import streamlit as st
import pandas as pd
import datetime

html_temp = """
<div style="background-color:lightblue;padding:5px">
<h1 style="color:black;text-align:center;">Where You At ⁉️🕐</h1>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.subheader('An app to predict absenteeism of an employee. Here is how it works :-')
st.text('1. Enter the below fields to the best of your knowledge.')
st.text("2. After filling all fields, click on the 'Go' button.")
st.text('3. The app will predict and tell whether the employee is Excessively absent or not.')
st.text('4. Enjoy ;)')
st.write("NOTE : An employee is considered to be 'Excessively Absent' if he/she is absent for more than 3 hours.")
st.text('')

reasons = ['1. Certain infectious and parasitic diseases',
           '2. Neoplasms',
           '3. Diseases of the blood and blood-forming organs abnd certain disorders involving the immune mechanism',
           '4. Endocrine, nutritional and metabolic diseases',
           '5. Mental and behavioural disorders',
           '6. Diseases of the nervous system',
           '7. Diseases of the eye and adnexa',
           '8. Diseases of the ear and mastoid process',
           '9. Diseases of the circulatory system',
           '10. Diseases of the respiratory system',
           '11. Diseases of the digestive system',
           '12. Diseases of the skin and subcutaneous tissue',
           '13. Diseases of the musculoskeletal system and connective tissue',
           '14. Diseases of the genitourinary system',
           '15. Pregnancy, childbirth and the puerperium',
           '16. Certain conditions originating in the perinatal period',
           '17. Congenital malformations, deformations and chromosomal abnormalities',
           '18. Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified',
           '19. Injury, poisoning and certain other consequences of external causes',
           '20. External causes of the morbidity and mortality',
           '21. Factors influencing health status and contact with the health services',
           '22. Patient follow-up',
           '23. Medical consultations',
           '24. Blood donation',
           '25. Laboratory examination',
           '26. Unjustified absence',
           '27. Physiotherapy',
           '28. Dental consultation'
           ]

education_list = ['1. High School',
                  '2. Graduate',
                  '3. Post Graduate',
                  '4. Master/Doctorate']

age = st.text_input('Enter age (in years) :')
reason = st.selectbox("Select the reason of absence : ", reasons)
date = st.date_input('Select date (YYYY/MM/DD):')
transport_expense = st.text_input('Enter transportation expense (in dollars):')
dist_to_work = st.text_input('Distance to work (in Kms):')
daily_work_load_avg = st.text_input('Enter daily work load average (in minutes):')
bmi = st.text_input('Enter Body Mass Index(BMI) :')
education = st.selectbox('Enter your education :', education_list)
children = st.text_input('Enter number of child/children :')
pets = st.text_input('Enter number of pet(s) :')

def format_input(age, reason, date, transport_expense, dist_to_work, daily_work_load_avg, bmi, education,
                 children, pets):
    reason_num = reason.split(' ')[0].split('.')[0]
    date = str(date)
    date_formatted = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%d/%m/%Y')
    education_num = education.split(' ')[0].split('.')[0]
    row = pd.DataFrame({'ID' : 1,
                        'Reason for Absence' : reason_num,
                        'Date' : date_formatted,
                        'Transportation Expense' : transport_expense,
                        'Distance to Work' : dist_to_work,
                        'Age' : age,
                        'Daily Work Load Average' : daily_work_load_avg,
                        'Body Mass Index' : bmi,
                        'Education' : education_num,
                        'Children' : children,
                        'Pets' : pets
                        }, index=[0])
    return row

if(st.button('Go')):
    df_row = format_input(age, reason, date, transport_expense, dist_to_work, daily_work_load_avg, bmi, education,
                 children, pets)
    classifier = absenteeism_model('model', 'scaler')
    classifier.load_and_clean_data_single_input(df_row)
    result = classifier.predicted_outputs()
    st.success('1. Probablity of employee being Excessively absent : %0.3f' % result['Probability'][0])
    if result['Prediction'][0] == 0:
        absent = 'NO'
    else:
        absent = 'YES'
    st.success('2. Will the employee be "Excessively absent" ? :::    {0}'.format(absent))