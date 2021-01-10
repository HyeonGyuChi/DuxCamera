from DuxCamera.ImgProcessing import basic as IPbasic
from DuxCamera.ImgProcessing import calculate as IPcalculate
from DuxCamera.ImgProcessing import calibration as IPcalibration

import matplotlib.pyplot as plt
import numpy as np
from object import CalculateObj, CalibrationObj

import cv2

if __name__ == "__main__":

    # IMG ROAD, 흑백
    q1_img = cv2.imread('./assets/source/cali_1.jpg', 0)
    q2_img = cv2.imread('./assets/source/cali_2.jpg', 0)
    q3_img = cv2.imread('./assets/source/cali_3.jpg', 0)
    q4_img = cv2.imread('./assets/source/cali_4.jpg', 0)
    
    # center확인
    plt.imshow(IPbasic.draw_center(q1_img), cmap='gray'); plt.show();

    # 이미지 이진화 및 closing연산
    q1_bin = IPbasic.img_closing(IPbasic.img_thresh(q1_img))
    q2_bin = IPbasic.img_closing(IPbasic.img_thresh(q2_img))
    q3_bin = IPbasic.img_closing(IPbasic.img_thresh(q3_img))
    q4_bin = IPbasic.img_closing(IPbasic.img_thresh(q4_img))

    plt.imshow(q1_bin); plt.show();

    # 각 쿼터 blob 정보
    q1_blob_info = IPcalibration.simpleBlobDetect(q1_bin)

    # find_centroid 이용해 무게중심 정보 가져오기 para = (이진화 이미지, blob info)
    q1_centroid = IPcalibration.find_centroid(q1_bin, q1_blob_info)

    ### calibration 정보 ###
    # centroid 좌표
    q1_centroid_point = q1_centroid[0]

    # centroid 사각형 w,h
    q1_centroid_square_shape = q1_centroid[1]

    # 각 쿼터 사각형 point 
    q1_x_max_blob_point, q1_y_min_blob_point, q1_x_min_blob_point, q1_y_max_blob_point = q1_centroid[2]
    
    # theta 구하기 4개의 꼭지점 구해 평균 각도 구하기(두점 사이의 각도)
    q1_x_max_theta = IPbasic.calc_theta(q1_centroid_point, q1_x_max_blob_point)# (center, target) (from, to) --> 45
    q1_y_min_theta = IPbasic.calc_theta(q1_centroid_point, q1_y_min_blob_point)# (center, target) (from, to) --> 135
    q1_x_min_theta = IPbasic.calc_theta(q1_centroid_point, q1_x_min_blob_point)# (center, target) (from, to) --> 225(-135)
    q1_y_max_theta = IPbasic.calc_theta(q1_centroid_point, q1_y_max_blob_point)# (center, target) (from, to) --> 315(-45)

    # 4개의 꼭지점이 틀어진 평균각도
    q1_theta = ((45 - q1_x_max_theta) + (135 - q1_y_min_theta) + (-135 - q1_x_min_theta) + (-45 - q1_y_max_theta))  / 4

    print(' --- theta info --- ')
    print(q1_x_max_theta, q1_y_min_theta, q1_x_min_theta, q1_y_max_theta)
    print(45 - q1_x_max_theta, 135 - q1_y_min_theta, -135 - q1_x_min_theta, -45 - q1_y_max_theta)


    print(' --- centoirds --- ')
    print(q1_centroid_point)

    print(' --- ref_square w,h --- ')
    print(q1_centroid_square_shape)
    print(' --- --------- --- ')

    print(' --- theta --- ')
    print(q1_theta)
    print(' --- ----- --- ')

    ### img merge ###
    # center
    center = IPbasic.get_crosspt(q1_x_min_blob_point, q1_x_max_blob_point, q1_y_min_blob_point, q1_y_max_blob_point)
    target = q1_x_max_blob_point

    

    print(center, target)

    print(IPbasic.calc_theta((int(center[0]), int(center[1])), target))

    