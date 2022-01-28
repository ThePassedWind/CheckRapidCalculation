import sys
import os
from streamlit import cli as stcli
import streamlit as st
import numpy as np
from PIL import Image,ImageFont,ImageDraw
import matplotlib.pyplot as plt
from yolo import YOLO
from crnn_master.hw4_2 import parse_opt, main
import shutil
from calculate import outcome
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2 as cv

# 评估用户输入图片的质量
def Evaluation(image):
    my_bar = st.progress(0)
    img = image.copy()
    # 转为opencv格式
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 1、放缩至相同尺寸412*412
    img = cv.resize(img, (412, 412), interpolation=cv.INTER_CUBIC)
    # 2、转为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3、计算清晰度，该值越大越好
    result = cv2.Laplacian(gray, cv2.CV_64F).var()
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1)
    st.write('清晰度：' + str(result))
    if result<1500:
        st.write('**您输入的图片过于模糊，请重新拍摄！**')
    else:
        st.write('**您输入的图片清晰度符合标准！**')

def ModelUpdate():
    st.write('ModelUpdate!!!')

def get4pos(box, image):
    top, left, bottom, right = box
    top = max(0, np.floor(top).astype('int32'))
    left = max(0, np.floor(left).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom).astype('int32'))
    right = min(image.size[0], np.floor(right).astype('int32'))
    return top, left, bottom, right

# 返回yolo框出的区域，并将其等式图片存入对应文件夹中
def GetBoxesPic(image, boxes):
    pics = []
    shutil.rmtree('./yolo3/tmp_img')
    os.mkdir('./yolo3/tmp_img')
    for i in range(len(boxes)):
        top, left, bottom, right = get4pos(boxes[i], image)
        pic = image.crop((left-15, top, right+40, bottom))
        # pic = image.crop((left, top, right, bottom))
        pic.save('./yolo3/tmp_img/pic'+str(i).rjust(3,'0')+'.jpg')
        pics.append(pic)
    return pics

# 进行yolo检测，呈现在web页面上
def Detecting(image):
    st.subheader("Detected Image")
    st.write("Just a second ...")
    yolo = YOLO()
    my_bar = st.progress(0)
    img = image.copy()
    start1 = time.time()
    r_image, boxes, top_conf = yolo.detect_image(image)
    end1 = time.time()
    print(r_image)
    for percent_complete in range(100):
        my_bar.progress(percent_complete + 1)
    st.image(r_image, use_column_width=True)
    st.subheader("Analysis Report")
    plt.scatter(np.arange(len(top_conf)), top_conf)
    plt.xlabel('detected rectangle')
    plt.ylabel('score')
    st.pyplot()
    # st.balloons()
    pics = GetBoxesPic(img, boxes)
    return boxes, end1-start1

def painting(equations, image, boxes):
    imgdraw = ImageDraw.ImageDraw(image)  # 创建一个绘图对象，传入img表示对img进行绘图操作
    font = ImageFont.truetype('simhei.ttf', image.size[1]//50, encoding="utf-8")
    for i in range(len(boxes)):
        top, left, bottom, right = get4pos(boxes[i], image)
        if outcome(equations[i]):
            imgdraw.text(xy=(left, bottom+3), text=equations[i]+'√', fill=(255,0,0), font=font)
        else:
            imgdraw.text(xy=(left, bottom + 3), text=equations[i] + '×', fill=(255, 0, 0),
                         font=font)  # 调用绘图对象中的text方法表示写入文字
    st.subheader("Recognition")
    st.image(image, use_column_width=True)

def detec_acc_analysis():

    # loss
    loss = go.Bar(
        x=['训练集', '验证集', '测试集'],
        y=[0.8488874537896973, 0.5683902647437119, 0.5204568483480593],
        name='loss'
    )
    acc = go.Scatter(
        x=['训练集', '验证集', '测试集'],
        y=[0.9835243553008596, 0.9664634146341463, 0.9024390243902439],
        name='acc'
    )
    layout = go.Layout(width=1000, height=500, yaxis1=dict(title='$温度（^oC）$'),
                       yaxis2=dict(title='湿度（%）', anchor='x', overlaying='y', side='right'),
                       xaxis=dict(title='时间'))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(loss)
    fig.add_trace(acc, secondary_y=True)
    fig['layout'].update(height=600, width=800, title='训练集/验证集/测试集的准确率和损失对比图',yaxis1=dict(title='loss'),
                         yaxis2=dict(title='acc(100%)'))
    st.subheader("Detection accuracy analysis")
    st.plotly_chart(fig)

def acc_analysis(equations, image, boxes):
    sum = len(boxes)
    count = 0
    for i in range(len(boxes)):
        top, left, bottom, right = get4pos(boxes[i], image)
        if outcome(equations[i]):
            count += 1
    st.subheader("Answer accuracy analysis")
    labels = ['答对数', '答错数']
    values = [count, sum - count]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    # fig.show()
    st.plotly_chart(fig)

def Time(detec_time, recog_time):
    st.subheader("Reasoning time analysis")
    labels = ['Time']
    fig = go.Figure(data=[
        go.Bar(name='检测时间', x=labels, y=[detec_time]),
        go.Bar(name='识别时间', x=labels, y=[recog_time])
    ])
    fig['layout'].update(height=600, width=800, yaxis1=dict(title='time(s)'))
    st.plotly_chart(fig)

def tt():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Handwriting Recognition")
    st.write("")
    file_up = st.file_uploader("Upload an image", type="jpg")

    if file_up is not None:
        image = Image.open(file_up)
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
        st.write("")
        img = image.copy()
        if st.button('Submit it'):
            st.write("Succeed!!!")
        if st.button('Evaluate the quality'):
            Evaluation(image)
        if st.button('Model Update'):
            ModelUpdate()
        if st.button('Detecting'):
            # 等式检测
            boxes, time1 = Detecting(image)
            # 文本识别
            my_bar = st.progress(0)
            start2 = time.time()
            opt = parse_opt()
            equations = main(opt)
            end2 = time.time()
            for percent_complete in range(100):
                my_bar.progress(percent_complete + 1)
            painting(equations, img, boxes)
            # print(equations)
            # 高级要求
            # 文本识别率识别分析
            detec_acc_analysis()

            # 用户答题准确率分析
            acc_analysis(equations, img, boxes)

            # 推理时间分析
            Time(time1, end2 - start2)

if __name__ == '__main__':

    if st._is_running_with_streamlit:
        tt()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())

