# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from yolo import YOLO
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Handwriting Recognition")
st.write("")
file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)
    st.write("")
    if st.button('Submit it'):
        st.subheader("Detected Image")
        st.write("Just a second ...")

        yolo = YOLO()
        video_path = 0
        video_save_path = ""
        video_fps = 25.0
        # -------------------------------------------------------------------------#
        #   test_interval用于指定测量fps的时候，图片检测的次数
        #   理论上test_interval越大，fps越准确。
        # -------------------------------------------------------------------------#
        test_interval = 100
        # -------------------------------------------------------------------------#
        #   dir_origin_path指定了用于检测的图片的文件夹路径
        #   dir_save_path指定了检测完图片的保存路径
        #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
        # -------------------------------------------------------------------------#
        my_bar = st.progress(0)
        r_image,boxes,top_conf = yolo.detect_image(image)

        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1)
        st.image(r_image, use_column_width=True)
        # st.download_button(label="Download image", data=r_image, file_name='large_df.jpg', mime="image/jpg")

        st.subheader("Analysis Report")
        plt.scatter(np.arange(len(top_conf)),top_conf)
        plt.xlabel('detected rectangle')
        plt.ylabel('score')
        st.pyplot()
        st.balloons()
        # with st.container():
        #     st.write("This is inside the container")
        #     # You can call any Streamlit command, including custom components:
        #     st.bar_chart(np.random.randn(50, 3))
        # st.write("This is outside the container")



