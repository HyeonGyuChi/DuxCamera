

import numpy as np
import cv2

### 맴버변수
# 각 쿼터 theta (공통) cali
# 각 쿼터 중앙 point (공통) cali

# 각 쿼터 이미지의 path (공통)
# 각 쿼터 이미지 객체 (공통)

# merge된 이미지 객체 (공통)

# 25개 blob 사각형 영역의 w,h (공통) cali
# 25개 blob의 point (공통) cali
# 25개 blob의 밝기 ()

### 메소드
## setter
# 각 쿼터 path와 이미지 객체 초기화
# cali정보 초기화(theta, centroids, 사각형영역w,h, 25 blob point)

## getter
# 25개 blob 밝기 가져오기


# 부모 class
class UnionObj :
    def __init__(self, q1_img_path=None, q2_img_path=None, q3_img_path=None, q4_img_path=None) :
        # for calibration
        self.q1_theta, self.q2_theta, self.q3_theta, self.q4_theta = (0,)*4
        self.q1_centroid, self.q2_centroid, self.q3_centroid, self.q4_centroid = (0,)*4

        # quarter img
        if q4_img_path :
            # 초기화
            self.q1_img_path, self.q2_img_path, self.q3_img_path, self.q4_img_path = q1_img_path, q2_img_path, q3_img_path, q4_img_path
            self.q1_img = cv2.imread(q1_img_path, 0) # 흑백으로 저장
            self.q2_img = cv2.imread(q2_img_path, 0) # 흑백으로 저장
            self.q3_img = cv2.imread(q3_img_path, 0) # 흑백으로 저장
            self.q4_img = cv2.imread(q4_img_path, 0) # 흑백으로 저장
            
        else : 
            self.q1_img_path, self.q2_img_path, self.q3_img_path, self.q4_img_path = ('',)*4
            self.q1_img, self.q2_img, self.q3_img, self.q4_img = (None,)*4
        
        
        # merged img
        self.merge_img = None

        # blob info
        self.blob_square_w, self.blob_square_h = (0,)*2
        self.blob_points = None # (None, (x,y)...)
        self.blob_brights = None # pandas


    def set_theta(self, q1_theta, q2_theta, q3_theta, q4_theta) :
        self.q1_theta, self.q2_theta, self.q3_theta, self.q4_theta = q1_theta, q2_theta, q3_theta, q4_theta

    def set_centorid(self, q1_theta, q2_theta, q3_theta, q4_theta) :
        self.q1_centroid, self.q2_centroid, self.q3_centroid, self.q4_centroid = q1_theta, q2_theta, q3_theta, q4_theta

    def get_blob_brights(self) :
        return self.blob_brights



# 자식 class(실제 이미지 분석용)
class CalculateObj(UnionObj) :
    def __init__(self, q1_img_path=None, q2_img_path=None, q3_img_path=None, q4_img_path=None) :
        super().__init__(q1_img_path, q2_img_path, q3_img_path, q4_img_path)



# 자식 class(Calibration용)
class CalibrationObj(UnionObj) :
    def __init__(self, q1_img_path=None, q2_img_path=None, q3_img_path=None, q4_img_path=None) :
        super().__init__(q1_img_path, q2_img_path, q3_img_path, q4_img_path)
    

    



