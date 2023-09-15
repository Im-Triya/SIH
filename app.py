import streamlit as st
import pickle
import numpy as np
import cv2 as cv
import caer

model=pickle.load(open('model1.pkl', 'rb'))

characters=['Betel Leaves',
 'Mint Leaves',
 'Balloon vine',
 'Amaranthus Green',
 'Coriander Leaves',
 'Curry Leaf',
 'Black Night Shade',
 'Malabar Spinach (Green)',
 'Giant Pigweed',
 'False Amarnath']

IMG_SIZE=[80, 80]

channels=1

#"D:\SIH new\Images\Curry1.jpg"


test_path = r'/kaggle/input/test-leaves/test_curry.webp'
img = cv.imread(test_path)
'''plt.imshow(img, cmap='gray')
plt.show()'''
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE,1)
    return img

predictions = model.predict(prepare(img))
# Getting class with the highest probability
print(characters[np.argmax(predictions[0])])

# def predict_sales(temperature):
#     input= np.array([[temperature]]).astype(np.float64)
#     prediction = model.predict(input)
#     pred = '{0:.{1}f}'.format(prediction[0][0], 2)
#     return float(pred)

def main():
    st.title("Sales Prediction")
    html_temp = """
    <div style="baackground-color:#025246" ; padding:10px>
    <h1>Sales Prediction Model </h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    temperature= st.text_input("Temperature", "Type Here")

    safe_html = """
    <div style="background-color:#F88888"; padding:10px>
    <h3 style="text-align:center"> Sales High</h3>
    </div>
    """

    danger_html = """
    <div style="background-color:#F88888"; padding:10px>
    <h3 style="text-align:center"> Sales Low</h3>
    </div>
    """

    if st.button("Predict"):
        output=predict_sales(temperature)
        st.success('The revenue generated will be {}'.format(output))

        if output>200:
            st.markdown(safe_html, unsafe_allow_html=True)
        else:
            st.markdown(danger_html, unsafe_allow_html=True)

if __name__=='__main__':
    main()
