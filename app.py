
import streamlit as st
# Disable PyplotGlobalUseWarning: You are calling st.pyplot() without any arguments. 
# After December 1st, 2020, we will remove the ability to do this as it requires the use of Matplotlib's global figure object, which is not thread-safe.
st.set_option('deprecation.showPyplotGlobalUse', False)
import json
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



st.title("Neural Network")
st.markdown("Input Image")


model = tf.keras.models.load_model('model.h5')
feature_model = tf.keras.models.Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

_, (x_test,_) = tf.keras.datasets.mnist.load_data()
x_test=x_test/255


def get_prediction(image):
    image_arr = np.reshape(image,(1,784))
    return feature_model.predict(image_arr), image



def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')


if st.button('Get Random Prediction'):
    index=np.random.choice(x_test.shape[0])
    image = x_test[index,:,:] 
    preds,image = get_prediction(image)
    final_preds = [p.tolist() for p in preds]
    image = np.reshape(image,(28,28))

    st.sidebar.image(image, width=150)
    for layer, p in enumerate(final_preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32,4))
        if layer ==2:
            row=1
            col=10
        else:
            row=2
            col=16

        for i,number in enumerate(numbers):
            plt.subplot(row, col,i+1)
            plt.imshow(number*np.ones((8,8,3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])
            if layer ==2:
                plt.xlabel(str(i),fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text('Layer {}'.format(layer+1))
        st.pyplot()
    
    
