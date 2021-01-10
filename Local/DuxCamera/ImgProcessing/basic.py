### 기본 이미지처리를 위한 모듈 ### 

import numpy as np

from skimage import img_as_ubyte
from skimage.filters import gaussian
from skimage.color import rgb2gray

import cv2

import matplotlib.pyplot as plt

import math
import os


# img gray변환
def img_gray(img) : 
    if len(img.shape) == 3 : # 칼라
        img = rgb2gray(img) # [0,1]로 반환
    
    img = img_as_ubyte(img) # [0,255]로 변환
    
    return img

# img 이진화
def img_thresh(img) : 
    if len(img.shape) == 3 : # 칼라
        img = rgb2gray(img)
    
    img = gaussian(img, sigma=0.5)
    
    img = img_as_ubyte(img)
    
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    return img_bin


# closing
def img_closing(img) : 

    # 팽창 후 침식(닫기), thresh로 제거된 tube내부 채워주기
    # 커널 생성
    # kernel = np.ones((5,5),np.uint8)
    if(img.dtype != np.uint8) : # opencv 팽창 침식을 위해 지원 데이터형으로 변경
        img = img.astype(np.uint8)
    
    
    kernel = np.array([[0,1,1,1,0],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [0,1,1,1,0],], np.uint8)
    # 팽창 후 침식
    dilate = cv2.dilate(img,kernel,iterations = 10)
    erode = cv2.erode(dilate,kernel,iterations = 10)

    # plt.figure(figsize=(30,50))
    # plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('origin_img')
    # plt.subplot(122), plt.imshow(erode, cmap='gray'), plt.title('closing')
    # plt.show()
    
    return erode

# opening
def img_opening(img) : 

    # 침식 후 팽창(열기), thresh로 제거된 tube내부 채워주기
    # 커널 생성
    # kernel = np.ones((5,5),np.uint8)
    if(img.dtype != np.uint8) : # opencv 팽창 침식을 위해 지원 데이터형으로 변경
        img = img.astype(np.uint8)
    
    
    kernel = np.array([[0,1,1,1,0],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [1,1,1,1,1],
                       [0,1,1,1,0],], np.uint8)
    # 침식 후 팽창
    # erode = cv2.erode(img,kernel,iterations = 10)
    # dilate = cv2.dilate(erode,kernel,iterations = 10)
    
    dilate = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 5)
    

    # plt.figure(figsize=(30,50))
    # plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('origin_img')
    # plt.subplot(122), plt.imshow(dilate, cmap='gray'), plt.title('opening')
    # plt.show()
    
    return dilate


# 중앙 그리기 함수
def draw_center(img, color=(255,255,255)) :
    
    r,g,b = color[0:3]
    
    img_copy = img.copy()
    img_h, img_w = img.shape[0:2]    
    
    # hline
    img_copy = cv2.line(img_copy, (int(img_w/2), 0), (int(img_w/2), img_h), (r,g,b), 10)
    
    # vline
    img_copy = cv2.line(img_copy, (0, int(img_h/2)), (img_w, int(img_h/2)), (r,g,b), 10)
    
    return img_copy

# 두점 사이의 거리 구하기 함수
def point2_distance(p1, p2) : # p1 = (x,y) # p2 = (x,y)
     
    p1_x, p1_y = p1 # 점1
    p2_x, p2_y = p2 # 점2
 
    a = p2_x - p1_x    # 선 a의 길이
    b = p2_y - p1_y    # 선 b의 길이
 
    c = math.sqrt((a * a) + (b * b))    # (a * a) + (b * b)의 제곱근을 구함 # 피타고라스
    
    return c

# 두 쌍 점(두 라인)사이의 교점구하기, 좀 이상함;
# line1(pa1, pa2)  : (x11,y11)-(x12,y12)  line2(pb1, pb2) : (x21,y21)-(x22,y22)
def get_crosspt(pa1, pa2, pb1, pb2) : 
    # line 1
    x11, y11 = pa1
    x12, y12 = pa2
    # line 2
    x21, y21 = pb1
    x22, y22 = pb2

    if x12==x11 or x22==x21:
        print('delta x=0')
        if x12==x11:
            cx = x12
            m2 = (y22 - y21) / (x22 - x21)
            cy = m2 * (cx - x21) + y21
            return cx, cy
        if x22==x21:
            cx = x22
            m1 = (y12 - y11) / (x12 - x11)
            cy = m1 * (cx - x11) + y11
            return cx, cy

    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1==m2:
        print('parallel')
        return None
    print(x11,y11, x12, y12, x21, y21, x22, y22, m1, m2)
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return (cx, cy)




# 두점 사이의 theta 구하는 함수, 좀 이상함;
def calc_theta(centroid_point, ref_point) : # (center, target) (from, to)
    h, w = (2448, 3264)
    demo_img = np.zeros((h, w), dtype=np.uint8)
    

    center_x, center_y = centroid_point # 중심
    ref_point_x, ref_point_y = ref_point # to
    
    cv2.circle(demo_img, (center_x,center_y),  1, (100, 100, 100), 70)
    cv2.circle(demo_img, (ref_point_x, ref_point_y),  1, (255, 255, 255), 35)
    cv2.line(demo_img, centroid_point, ref_point, (175, 175, 175), 10)
    
    # center 선
    cv2.line(demo_img, (center_x, 0), (center_x, h), (175, 175, 175), 5)
    cv2.line(demo_img, (0, center_y), (w, center_y), (175, 175, 175), 5)
    
    plt.figure(figsize=(10,10));
    plt.imshow(demo_img, cmap='gray'); plt.title('Center - Target Point'); plt.show();
    
    # 각도 구하기
    dx = ref_point_x - center_x
    dy = ref_point_y - center_y
    
    radian = math.atan2(dy, dx)
    degree = -(radian  * 180  / math.pi)
    
    ''' 이미지와 평면좌표가 다름, y축이 반대, 그러므로 연산각도도 - 로 변경됨
    3 | 4
    2 | 1 
    '''
    return degree 

