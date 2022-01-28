
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from yolo import YOLO

img = Image.open("./img/street.jpg")
# img.show()
# img.crop((100,200,400,500)).save('./img/street2.jpg')

