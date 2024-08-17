import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("trained_model.sav", 'rb'))

def insur(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    print('The insurance cost is USD ', prediction[0])

def main():
    st.title('Insurance')
    a=st.text_input('Age')
    d = st.text_input('Sex')
    b=st.text_input('Bmi')
    c=st.text_input('Children')
    e=st.text_input('Smoker')
    f=st.text_input('Region')

    charges = ''
    if st.button('Predict'):
        charges=insur([a,d,b,c,e,f])
        st.success(charges)

if __name__ == '__main__':
    main()


