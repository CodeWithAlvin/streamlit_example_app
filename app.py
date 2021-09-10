import streamlit as st
import numpy as np
import pandas as pd
from backend import load_model,predict
from PIL import Image
import matplotlib.pyplot as plt

# LOADING MODEL
model_path = 'model_densenet_efnet_vgg_09581.h5'

model = load_model(model_path)


#adding custom css
with open("src/css/app.css","r") as css:
    st.markdown("""<style>{}</style>""".format(css.read()),unsafe_allow_html=True)

# design and style for the navbar
st.markdown(
    """
    <style>
    [data-testid="Navbar"][aria-expanded="true"] > div:first-child{
        width: 350px
    } 
    [data-testid="Navbar"][aria-expanded="false"] > div:first-child{
        width: 0px
        overflow:hidden
    } 
    </style>
    """,unsafe_allow_html=True
)

st.sidebar.image("src/img/about.jpg",width=64)
st.sidebar.title("ML for Health")# setting title on navbar

# defining available app mode 
app_mode = st.sidebar.selectbox(
    "",
    ["Make Predictions","About App"]
)

# desigining about section
if app_mode == "About App":
    _, mid, _ = st.columns([1.5, 1, 1.5])
    with mid:
        st.title('About Me')

    _, mid2, _ = st.columns([1, 1, 1])
    with mid2:
        st.markdown("<br/><br/>",unsafe_allow_html=True)
        st.image("src/img/about.jpg",width=200)
        st.markdown("<br/><br/>",unsafe_allow_html=True)
    
    
    st.markdown("""
    I’m a **Full-Stack Web Developer**, and **Machine Learning Enthusiast**, located in India. I use to spend my free time Learning and applying Different Machine/Deep Learning algorithms to make life Easy.

    I am in web Development since October 2020. Except programming, I'm passionate about Robotics Aeronautics, and Electronics.

    Currently, I am exploring Neural Networks and Augmented Reality.
    """)
    
elif app_mode == "Make Predictions":
    st.title('Automated Diagnosis of the Diseases Pneumonary Tuberculosis')
    st.markdown("<br/><br/>",unsafe_allow_html=True)
    st.markdown("### Upload an Image")
    img_file_buffer = st.file_uploader("",type=["jpg","jpeg","png"])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        _, img_grid, _ = st.columns([0.2,1,0.2])
        with img_grid:
            st.image(image,width=500)
            btn = st.button('Predict Result')

        if btn:
            st.markdown("---")
            # call the function here
            normal,pneumonia,tuberculosis = predict(image, model)
            result = None

            if pneumonia > normal and pneumonia > tuberculosis:
                result = "PNEUMONIA"
            elif normal > pneumonia  and normal > tuberculosis:
                result = "NORMAL"
            else:
                result = "TUBERCULOSIS"

            
            grid_1,grid_2 = st.columns([1,1])
        
            with grid_1:
                st.markdown ("<h1 class='color-heading' >Metrics</h1>",unsafe_allow_html=True)
            
            with grid_2:
                st.markdown("<h1 class='color-heading'>Prediction</h1> ", unsafe_allow_html=True )
            
            grid_res1,grid_res2 = st.columns ([1,1])

            with grid_res1:
                metrics = pd.DataFrame(
                    [["Pneumonia",pneumonia],
                    ["tuberculosis",tuberculosis],
                    ["normal",normal]]
                ,columns=['disease','accuracy'])
                
                st.table(metrics)
            
            with grid_res2:
                st.markdown(f"## {result}")

            st.markdown("---")
            fig1, ax1 = plt.subplots()
            ax1.pie(metrics["accuracy"],labels=metrics["disease"], autopct='%1.1f%%',
                            shadow=True, startangle=90)
            ax1.axis('equal')   
            st.pyplot(fig1)
            