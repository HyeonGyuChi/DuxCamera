### Calibration과 이미지 Merge에 사용되는 모듈 ### 

import numpy as np

from PIL import Image

# from skimage import color, viewer, img_as_float, img_as_ubyte, img_as_uint, data
# from skimage.filters import gaussian
# from skimage.color import rgb2gray

# import matplotlib.image as mpimg

import matplotlib.pylab as plt

import seaborn as sns

import os

import cv2

from DuxCamera.ImgProcessing.basic import point2_distance

# BLOB 찾는 함수
# https://www.learnopencv.com/blob-detection-using-opencv-python-c/
# https://www.theteams.kr/teams/7191/post/70373
# https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html?highlight=blob
def simpleBlobDetect(img) :
    img_copy = np.copy(img)
    blob_info = [] # 추출한 blob 정보
    
    # blob detection
    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 255 # 밝은 얼룩 추출

    # params.minThreshold = 240
    # params.maxThreshold = 255

    params.filterByArea = True
    params.minArea = 10*10;
    params.maxArea = 200*200

    params.filterByCircularity = True
    params.minCircularity = 0.8;
    # 원 = 1.0
    # 사각형 = 0.785
    

    params.filterByConvexity = False

    params.filterByInertia = True
    params.minInertiaRatio = 0.7;
    # 타원~원 = 0~1
    # 줄 = 0

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_copy)
    print('Detecting한 Blob개수 : ', len(keypoints))

    # Blob labeling 수행
    im_with_keypoints = cv2.drawKeypoints(img_copy, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for k in keypoints :
        x, y = k.pt
        x,y = int(x), int(y)

        print(k.pt, k.size,k.class_id) # 추출결과의 중심, 추출결과의 diameter (blob의 직경x)
        cv2.circle(img_copy, (x,y),  1, (155, 155, 155), 10)
        cv2.circle(img_copy, (x,y),  int(k.size/2), (155, 155, 155), 10)
        
        blob_info.append([x,y,k.size]) # x,y, diameter 정보
    
    blob_info = np.array(blob_info) # argmin, argmx 를 위해 numpy 사용
    
    plt.figure(figsize=(15,15))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('origin_binary_img')
    plt.subplot(122), plt.imshow(img_copy, cmap='gray'),  plt.title('Blob info')
    plt.show();
    
    return blob_info



# 무게중심(CX,CY) 찾기, 및 BLOB 사각형의 'ㄱ'부분 길이 구하기 (ref_square_w, ref_square_h), 
def find_centroid(img, blob_info) : 
    img_h, img_w = np.shape(img)
    img_copy = np.copy(img) # 무게중심 표시를 위한 img
    img_temp = np.zeros((img_h, img_w), dtype=np.uint8) # 25개의 blob중 가장자리 blob 으로 무게중심 찾기 위한 사각형 
    
    x_min_blob = blob_info[np.argmin(blob_info[::, 0])] # 모든 x에서 가장 작은 blob
    x_max_blob = blob_info[np.argmax(blob_info[::, 0])]
    y_min_blob = blob_info[np.argmin(blob_info[::, 1])] # 모든 x에서 가장 작은 blob
    y_max_blob = blob_info[np.argmax(blob_info[::, 1])]
    
    # int로 변경
    x_min_blob = x_min_blob.astype(np.int)
    x_max_blob = x_max_blob.astype(np.int)
    y_min_blob = y_min_blob.astype(np.int)
    y_max_blob = y_max_blob.astype(np.int)
    
    print('x_min_blob : ', x_min_blob[0:2])
    print('x_max_blob : ', x_max_blob[0:2])
    print('y_min_blob : ', y_min_blob[0:2])
    print('y_max_blob : ', y_max_blob[0:2])
    
    # side blob point 표시
    # cv2.circle(img_temp, (x_min_blob[0], x_min_blob[1]),  1, (155, 155, 155), 10)
    # cv2.circle(img_temp, (x_max_blob[0], x_max_blob[1]),  1, (155, 155, 155), 10)
    # cv2.circle(img_temp, (y_min_blob[0], y_min_blob[1]),  1, (155, 155, 155), 10)
    # cv2.circle(img_temp, (y_max_blob[0], y_max_blob[1]),  1, (155, 155, 155), 10)
    
    # 해당 side 포인트이 꼭지점을 이루는 사각형 그리기
    pts = np.array([[x_max_blob[0],x_max_blob[1]], 
                    [y_min_blob[0],y_min_blob[1]],
                    [x_min_blob[0],x_min_blob[1]],
                    [y_max_blob[0],y_max_blob[1]]], np.int32)
    
    pts = pts.reshape((-1,1,2))
    
    cv2.polylines(img_copy, [pts], isClosed = True, color = (155, 155, 155), thickness = 10) # 사각형 그리기
    cv2.fillPoly(img_temp, [pts], (155, 155, 155), cv2.LINE_AA) # 채워진 사각형 그리기
    # cv2.fillPoly(img_copy, [pts], (155, 155, 155), cv2.LINE_AA) # 채워진 사각형 그리기
    
    # img_temp의 무게중심 구하기
    contours, hierarchy = cv2.findContours(img_temp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    for i in contours:
        M = cv2.moments(i)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    
        cv2.circle(img_temp, (cX, cY), 15, (100, 100, 100), -1)
        cv2.circle(img_copy, (cX, cY), 15, (100, 100, 100), -1)
        cv2.drawContours(img_temp, [i], 0, (100, 100, 100), 10)
    
    print('Centroid : ', cX, cY)
    
    # ref_square에 구하기 'ㄱ'부분 길이 구하기
    ref_square_w = point2_distance(y_min_blob[0:2], x_max_blob[0:2]) # 'ㄱ'의 'ㅡ'부분
    ref_square_h = point2_distance(y_max_blob[0:2], x_max_blob[0:2]) # 'ㄱ'의 '|'부분
    
    print('ref_square_w : ', ref_square_w)
    print('ref_square_h : ', ref_square_h)
    
    plt.figure(figsize=(20,10))
    plt.subplot(121), plt.imshow(img_copy, cmap='gray'), plt.title('Centroid Point')
    plt.subplot(122), plt.imshow(img_temp, cmap='gray'), plt.title('Ref square from Side_Blob')
    plt.show();
    
    return ((int(cX), int(cY)), (int(ref_square_w), int(ref_square_h)),  ((x_max_blob[0], x_max_blob[1]), (y_min_blob[0], y_min_blob[1]), (x_min_blob[0], x_min_blob[1]), (y_max_blob[0], y_max_blob[1]))) # 25개 blob의 무게중심(cX, cY) # ref 사각형의 w,h # 사각형의 네 꼭지점 정보
    # return[0] = (cX, cY) # 25 blob 사각형의 무게중심
    # return[1] = (ref_square_w, ref_square_h) # 해당 사각형의 w,h
    # return [2] = (xmax, ymin, xmin, ymax) # ref 사각형의 w,h # 사각형의 네 꼭지점 정보




# 지정한 중점에서 theta만큼 이미지 돌리기
def img_affine(img, centroid, theta) : #### aFFINE 시 0~255 -> 0~1로 변경 
    img_copy = np.copy(img)
    
    # 회전하기 전에 center 표시 하여 얼마나 돌아갔는지 확인
    img_copy = cv2.circle(img_copy, centroid,  1, (220, 220, 0), 30)
    img_copy = cv2.putText(img_copy, 'theta = ' + str(theta), centroid, cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255),  cv2.LINE_AA)
    
    # 회전 opcn cv
    '''
    img_h, img_w = img.shape[0:2]
    # matrix = cv2.getRotationMatrix2D((img_w/2, img_h/2), theta, 1)
    matrix = cv2.getRotationMatrix2D(centroid, theta, 1)
    dst = cv2.warpAffine(img, matrix, (img_w, img_h)) # 0~1로 변경됨
    '''
    
    # 회전 pil
    img_h, img_w = img.shape[0:2]
     
    # pil 객체로 변경
    dst = Image.fromarray(img.astype('uint8'), 'L')
    dst = dst.rotate(theta, center=centroid, expand=False, resample=Image.NEAREST) # theta만큼 회전
    
    # 다시 numpy로 변경
    dst = np.array(dst)
    
    plt.figure(figsize=(10,10))
    plt.subplot(121), plt.imshow(img_copy, cmap='gray'), plt.title('Before_affine')
    plt.subplot(122), plt.imshow(dst, cmap='gray'), plt.title('After_affine')
    plt.show();
    
    print('img. max : ', np.max(np.unique(img)), 'img. min : ', np.min(np.unique(img)))
    print('affine img. max : ', np.max(np.unique(dst)), 'affine img. min : ', np.min(np.unique(dst)))
    
    return dst




# centroid를 중심으로 돌린 이미지를 설정한 기준으로 hw를 잘라 합성하기 위한 이미지를 만드는 함수
def img_cutting(img, centroids, shape = 1500) : # 2000
    img_copy = np.copy(img)
    
    result_h, result_w = shape, shape
    center_x, center_y = centroids
    
    # 시작 인덱스
    start_x = center_x - int(result_w / 2)
    start_y = center_y - int(result_h / 2)
    
    result = img_copy[start_y : start_y+result_h, start_x : start_x+result_w, ...]
    print(result.shape)
    
    return result




# 4개의 쿼터 이미지 지정한 center와 theta로 affine하여 merge된 완성된 이미지 추출
def img_merge(img_list, centroids_list, theta_list) : # 1,2,3,4,로 정렬된 각 쿼터 img list, 각 쿼터 center정보, 각 쿼터 theta정보    
    
    q1_img, q2_img, q3_img, q4_img = img_list
    q1_theta, q2_theta, q3_theta, q4_theta = theta_list
    q1_centroid, q2_centroid, q3_centroid, q4_centroid = centroids_list
    
    
    print('before merge img range = [ {} {}]'.format(np.unique(q1_img)[0], np.unique(q1_img)[-1])) # q1 unique range 정보
    
    q1_affine_img = img_affine(q1_img, q1_centroid, q1_theta)
    q2_affine_img = img_affine(q2_img, q2_centroid, q2_theta)
    q3_affine_img = img_affine(q3_img, q3_centroid, q3_theta)
    q4_affine_img = img_affine(q4_img, q4_centroid, q4_theta)
    
    
    q1_cut = img_cutting(q1_affine_img, q1_centroid)
    q2_cut = img_cutting(q2_affine_img, q2_centroid)
    q3_cut = img_cutting(q3_affine_img, q3_centroid)
    q4_cut = img_cutting(q4_affine_img, q4_centroid)
    
    
    # zeros 생성않하고 할 경우 uint8로 되어 [0-255]만 저장되어 255이상 값 overflow로 처리됨 => np.float32로 변경
    merged_img = q1_cut.astype(np.float32) + q2_cut.astype(np.float32) + q3_cut.astype(np.float32) + q4_cut.astype(np.float32)
    
    print('after merge img range = [ {} {}]'.format(np.unique(merged_img)[0], np.unique(merged_img)[-1])) # merged range 정보
    
    return merged_img