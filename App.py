import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

rf_model = joblib.load('Data\\RandomForest_best_model.pkl')
lr_model = joblib.load('Data\\LogisticRegression_best_model.pkl')
svm_model = joblib.load('Data\\SVM_best_model.pkl')
nn_model = load_model('Data\\NN_best_model.keras')
scaler = joblib.load('Data\\scaler.pkl')

st.title('Прогноз відтоку клієнта')

st.write('Введіть параметри клієнта, щоб передбачити ймовірність його відтоку.')

model_choice = st.selectbox('Виберіть модель для передбачення', ['Випадковий ліс', 'Логістична регресія', 'SVM', 'Нейронна мережа'])

is_tv_subscriber = st.selectbox('Чи є клієнт підписником телебачення?', [0, 1], format_func=lambda x: 'Так' if x == 1 else 'Ні')
is_movie_package_subscriber = st.selectbox('Чи є клієнт підписником кінопакету?', [0, 1], format_func=lambda x: 'Так' if x == 1 else 'Ні')
subscription_age = st.number_input('Вік підписки клієнта')
bill_avg = st.number_input('Середня сума рахунку клієнта')
service_failure_count = st.number_input('Кількість збоїв у наданні послуг клієнту')
download_avg = st.number_input('Середній обсяг завантажень клієнта')
upload_avg = st.number_input('Середній обсяг відтяжок клієнта')
download_over_limit = st.number_input('Кількість перевищень ліміту завантажень')

if st.button('Передбачити відтік'):
    data_to_scale = np.array([[subscription_age, bill_avg, service_failure_count, download_avg, upload_avg]])
    input_data = np.array([[is_tv_subscriber, is_movie_package_subscriber]])
    scaled_data = scaler.transform(data_to_scale)
    download_over_limit_array = np.array([[download_over_limit]])

    combined_data = np.hstack((input_data, scaled_data, download_over_limit_array))

    if model_choice == 'Випадковий ліс':
        churn_probability = rf_model.predict_proba(combined_data)[0][1] * 100
    elif model_choice == 'Логістична регресія':
        churn_probability = lr_model.predict_proba(combined_data)[0][1] * 100
    elif model_choice == 'SVM':
        churn_probability = svm_model.predict_proba(combined_data)[0][1] * 100
    elif model_choice == 'Нейронна мережа':
        churn_probability = nn_model.predict(combined_data)[0][0] * 100

    st.write(f"Ймовірність відтоку: {churn_probability:.2f}%")
    
    if churn_probability > 50:
        st.error("Ймовірність відтоку висока!")
    else:
        st.success("Ймовірність відтоку низька.")