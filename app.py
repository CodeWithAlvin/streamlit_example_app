import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# local file imports
from backend import load_model,predict
import text 

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

# setting logo image in navbar 
st.sidebar.image("src/img/logo.jpg",width=64)
st.sidebar.title(text.sidebar_title)# setting title on navbar

# defining available app mode 
app_mode = st.sidebar.selectbox(
    "",
    ["Make Predictions","About App"]
)

# adding a partian
st.sidebar.markdown("---")

# connect me section here present on Navbar
st.sidebar.markdown("**Connect me On**")
st.sidebar.markdown(f"""
![insta](https://img.icons8.com/ios/50/000000/instagram-new--v1.png)[instagram]
[instagram]:https://instagram.com/{text.instagram}

![twitter](https://img.icons8.com/ios/50/000000/twitter--v1.png)[twitter]
[twitter]:https://twitter.com/{text.twitter}

![linkedin](https://img.icons8.com/ios/50/000000/linkedin.png)[linkedin]
[linkedin]:https://linkedin.com/{text.linkedin}

""")

# desigining about section
if app_mode == "About App":
    # making grid to center text
    _, about_mid, _ = st.columns([1.5, 1, 1.5])
    with about_mid:
        st.title('About Me')

    #  making grid to center image
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        st.markdown("<br/><br/>",unsafe_allow_html=True)
        st.image("src/img/about.jpg",width=200)
        st.markdown("<br/><br/>",unsafe_allow_html=True)
    
    # setting about text
    st.markdown(text.about_text)

# if app mode is prediction which is by default
elif app_mode == "Make Predictions":
    # setting title
    st.title(text.homepage_title)
    st.markdown("<br/><br/>",unsafe_allow_html=True)
    
    # uploading image
    st.markdown("### Upload an Image")
    img_file_buffer = st.file_uploader("",type=["jpg","jpeg","png"])

    # procedding for prediction if image is not none
    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))
        _, img_grid, _ = st.columns([0.2,1,0.2])
        with img_grid:
            st.image(image)
            # adding button for prediction 
            btn = st.button('Predict Result')

        # if prediction button is pressed then calling the predict function from backend to predict the result
        if btn:
            st.markdown("---")
            # call the function here
            normal,pneumonia,tuberculosis = predict(image, model)
            result = None

            # checking which have greater probablity greater 
            if pneumonia > normal and pneumonia > tuberculosis:
                result = "PNEUMONIA"
            elif normal > pneumonia  and normal > tuberculosis:
                result = "NORMAL"
            else:
                result = "TUBERCULOSIS"

            # columns for showing metrics and results 
            grid_1, _, grid_2 = st.columns([1,0.2,1])
        
            with grid_1:
                st.markdown ("<h1 class='color-heading' >Metrics</h1>",unsafe_allow_html=True)
            
            with grid_2:
                st.markdown("<h1 class='color-heading'>Prediction</h1> ", unsafe_allow_html=True )
            
            grid_res1, _, grid_res2 = st.columns ([1,0.2,1])

            with grid_res1:
                # making Dataframe from data recived from model to show
                metrics = pd.DataFrame(
                    [["Pneumonia",pneumonia],
                    ["tuberculosis",tuberculosis],
                    ["normal",normal]]
                ,columns=['disease','accuracy'])
                
                st.table(metrics)
            
            # showing the result 
            with grid_res2:
                st.markdown(f"## {result}")

            st.markdown("---")

            # plotting pie chart on the data recived
            _,graph,_ = st.columns([0.1,0.8,0.1])
            with graph:
                fig1, ax1 = plt.subplots()
                ax1.pie(metrics["accuracy"],labels=metrics["disease"], autopct='%1.1f%%',
                            shadow=True, startangle=90)
                ax1.axis('equal')   
                st.pyplot(fig1)
            