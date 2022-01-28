# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
# import os
# import sys
# curPath = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(curPath)
# sys.path.append('D:\\pythonWorkPlace\\pycharmCodes\\curriculums\\Deep Learning\\crnn_master')
# print(sys.path)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from yolo import YOLO
# from crnn_master.hw4_2 import parse_opt, main
# # import importlib
# # res = importlib.import_module('crnn-master.detect')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Handwriting Recognition")
st.write("")
file_up = st.file_uploader("Upload an image", type="jpg")

def Labeling():
    st.write("labeling!!!")

def ModelUpdate():
    st.write('ModelUpdate!!!')

# 返回yolo框出的区域，并将其等式图片存入对应文件夹中
def GetBoxesPic(image, boxes):
    pics = []
    for i in range(len(boxes)):
        top, left, bottom, right = boxes[i]
        top = max(0, np.floor(top).astype('int32'))
        left = max(0, np.floor(left).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom).astype('int32'))
        right = min(image.size[0], np.floor(right).astype('int32'))
        pic = image.crop((left, top, right, bottom))
        pic.save('./tmp_img/pic'+str(i)+'.jpg')
        pics.append(pic)
    return pics

# 进行yolo检测，呈现在web页面上
def Detecting(img):
    st.subheader("Detected Image")
    st.write("Just a second ...")
    yolo = YOLO()
    my_bar = st.progress(0)
    r_image, boxes, top_conf = yolo.detect_image(image)
    print(r_image)
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1)
    st.image(r_image, use_column_width=True)
    # st.download_button(label="Download image", data=r_image, file_name='large_df.jpg', mime="image/jpg")
    st.subheader("Analysis Report")
    plt.scatter(np.arange(len(top_conf)), top_conf)
    plt.xlabel('detected rectangle')
    plt.ylabel('score')
    st.pyplot()
    st.balloons()
    pics = GetBoxesPic(img, boxes)
    # st.image(pic, use_column_width=True)


# 类似于主函数
if file_up is not None:
    image = Image.open(file_up)
    img = image.copy()
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)
    st.write("")
    if st.button('Submit it'):
        st.write("Succeed!!!")
    if st.button('Labeling'):
        Labeling()
    if st.button('Model Update'):
        ModelUpdate()
    if st.button('Detecting'):
        Detecting(img)
    # opt = parse_opt()
    # main(opt)