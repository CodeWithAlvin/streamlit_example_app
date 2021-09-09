import streamlit as st
import numpy as np
import cv2
from PIL import Image

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

st.sidebar.image("about.jpg",width=64)
st.sidebar.title("ML for Health")# setting title on navbar

@st.cache()
def image_resize(img,width=None,height=None):
    """
    stroing the height and width of old image 
    then adjusting aspect ratio 
    resizing image
    returning image
    """
    dim = None 
    h,w = img.shape[:2]

    if height is None and width is None:
        return img
    
    elif width is None:
        ratio = width/float(w)
        dim = (int(w*ratio),height)
    
    else:
        ratio = height/float(h)
        dim = (width,int(h*ratio))

    img = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    
    return img
     
# defining available app mode 
app_mode = st.sidebar.selectbox(
    "",
    ["Make Predictions","About App"]
)


# desigining about section
if app_mode == "About App":
    col1, mid, col2 = st.columns([1.5, 1, 1.5])
    with mid:
        st.title('About Me')

    col3, mid2, col4 = st.columns([1, 1, 1])
    with mid2:
        st.markdown("<br/><br/>",unsafe_allow_html=True)
        st.image("about.jpg",width=200)
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
        st.image(image)
        
        if st.button('Predict Result'):

            # call the function here
            normal,pneumonia,tuberculosis = 0.5, 0.3, 0.2
            result = None

            if pneumonia > normal and pneumonia > tuberculosis:
                result = "PNEUMONIA"
            elif normal > pneumonia  and normal > tuberculosis:
                result = "NORMAL"
            else:
                result = "TUBERCULOSIS"
            
    # TODO : implemnet ui for result