### BLOB의 밝기측정과 BLOB Point위치를 부여하는 모듈 ### 

# blob ref square 와 centroid를 활용하여 blob들의 points 위치 계산
# (이미지, centroid_point = ref square중앙(== moddle point), ref_square_shape = 존재하는 blob영역 사각형의 w,h)
def calc_BlobsPoint(img, centroid_point, ref_square_shape):
    print('\n┌━━━ Call func, [mean_bright] ━━━-┐')

    q_point = [0] * 26  # quarter points 정보
    cX, cY = centroid_point  # center정보
    ref_square_w, ref_square_h = ref_square_shape  # blob square ref 크기

    interval_w, interval_h = int(ref_square_w / 4), int(ref_square_w / 4)  # blob들 사이 간격

    # v line 정보 (y point)
    vline_1 = cY - (interval_h * 2)
    vline_2 = cY - (interval_h * 1)
    vline_3 = cY  # center y line
    vline_4 = cY + (interval_h * 1)
    vline_5 = cY + (interval_h * 2)

    # h line 정보 (x point)
    hline_1 = cX - (interval_w * 2)
    hline_2 = cX - (interval_w * 1)
    hline_3 = cX  # center x line
    hline_4 = cX + (interval_w * 1)
    hline_5 = cX + (interval_w * 2)

    # blob point 부여
    # vline_1 (첫번째 줄 blob)
    q_point[1] = [hline_1, vline_1]
    q_point[2] = [hline_2, vline_1]
    q_point[3] = [hline_3, vline_1]
    q_point[4] = [hline_4, vline_1]
    q_point[5] = [hline_5, vline_1]

    # vline_2 (두번째 줄 blob)
    q_point[6] = [hline_1, vline_2]
    q_point[7] = [hline_2, vline_2]
    q_point[8] = [hline_3, vline_2]
    q_point[9] = [hline_4, vline_2]
    q_point[10] = [hline_5, vline_2]

    # vline_3 (세번째 줄 blob)
    q_point[11] = [hline_1, vline_3]
    q_point[12] = [hline_2, vline_3]
    q_point[13] = [hline_3, vline_3]  # center blob(== cX, cY)
    q_point[14] = [hline_4, vline_3]
    q_point[15] = [hline_5, vline_3]

    q_point[16] = [hline_1, vline_4]
    q_point[17] = [hline_2, vline_4]
    q_point[18] = [hline_3, vline_4]
    q_point[19] = [hline_4, vline_4]
    q_point[20] = [hline_5, vline_4]

    q_point[21] = [hline_1, vline_5]
    q_point[22] = [hline_2, vline_5]
    q_point[23] = [hline_3, vline_5]
    q_point[24] = [hline_4, vline_5]
    q_point[25] = [hline_5, vline_5]

    # point 확인
    # 해당부분 확인을 위한 그리기 img
    img_copy = np.copy(img)

    max_color = np.int(np.max(img_copy))  # 이미지의 최대 색깔

    for i, (x, y) in enumerate(q_point[1:], 1):  # 1번 blob부터 찍기, [0]은 아무정보 없음
        print(i, 'blob points = ', x, y)
        cv2.circle(img_copy, (x, y), 3, (max_color, max_color, max_color), 10)

    plt.figure(figsize=(15, 15))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('origin_img')
    plt.subplot(122), plt.imshow(img_copy, cmap='gray'), plt.title('Blob points')
    plt.show();

    return q_point


# 1500x1500 img에서 25개 points 기준 밝기 구하기 함수
# degree = points 기준으로 얼만큼 체크할지 정도
# q_point = 각 25개 blob point정보 len =26개 ()
def calc_blob_MeanBright(img, q_point_info, degree=1):
    print('\n┌━━━ Call func, [calc_blob_MeanBright] ━━━-┐')

    tube_mean_bright = np.zeros([5, 5])  # 각 tube mean bright 정보
    h, w = tube_mean_bright.shape[0:2]
    q_idx = 1

    '''
    q_point=[0]*26 # quarter points 정보

    alpha = 250
    # 1 quater
    q_point[19]= [alpha+ 700,alpha+ 700]
    q_point[20] = [alpha+ 900,alpha+ 700]
    q_point[24] = [alpha+ 700,alpha+ 900]
    q_point[25] = [alpha+ 900,alpha+ 900]

    # 2 quater
    q_point[4]= [alpha+ 700,alpha+ 100]
    q_point[5] = [alpha+ 900,alpha+ 100]
    q_point[9] = [alpha+ 700,alpha+ 300]
    q_point[10] = [alpha+ 900,alpha+ 300]

    # 3 quater
    q_point[1] = [alpha+ 100,alpha+ 100]
    q_point[2] = [alpha+ 300,alpha+ 100]
    q_point[6] = [alpha+ 100,alpha+ 300]
    q_point[7] = [alpha+ 300,alpha+ 300]

    # 4 quater
    q_point[16] = [alpha+ 100,alpha+ 700]
    q_point[17] = [alpha+ 300,alpha+ 700]
    q_point[21] = [alpha+ 100,alpha+ 900]
    q_point[22] = [alpha+ 300,alpha+ 900]

    # middle point
    q_point[13] = [alpha+ 500,alpha+ 500]
    # q_point[13] = [centroid_point]

    # vline
    q_point[3] = [alpha + 500,alpha + 100]
    q_point[8]= [alpha + 500,alpha + 300]
    q_point[18] = [alpha + 500,alpha + 700]
    q_point[23] = [alpha + 500,alpha + 900]

    # hline
    q_point[11] = [alpha + 100,alpha + 500]
    q_point[12] = [alpha + 300,alpha + 500]
    q_point[14] = [alpha + 700,alpha + 500]
    q_point[15] = [alpha + 900,alpha + 500]
    '''

    # 해당부분 확인을 위한 그리기 img
    img_copy = np.copy(img)

    # img shape 확인
    img_h, img_w = img_copy.shape
    print('IMG SHAPE : (h : {}, w : {})'.format(img_h, img_w))

    # 표시 color // img 구성 pixel의 90%
    color = int(np.unique(img)[int(len(np.unique(img)) * 0.9)])
    display_color = (color, color, color)

    for i in range(h):  # 5
        for j in range(w):  # 5

            # degree 조정
            x, y = q_point_info[q_idx]
            left_x, left_y = x - degree, y - degree
            right_x, right_y = x + degree, y + degree

            # 평균밝기 연산
            roi = img[left_y: right_y + 1, left_x: right_x + 1]  # 밝기 참조영역
            mean_bright = np.mean(roi)  # 평균밝기(quarter4개를 더했으므로)

            # 확인
            # plt.subplot(5,5,q_idx), plt.imshow(roi, cmap='gray', vmin=0, vmax=255), plt.title(str(q_idx) + ' of ROI') # plt로 그리기
            # plt.subplot(5,5,q_idx), sns.heatmap(roi, annot=True, fmt=".2f", cmap='gray', vmin = 0, vmax = 255*4), plt.title(str(q_idx) + ' of ROI') # sns으로 heatmap 그리기

            img_copy = cv2.rectangle(img_copy, (left_x, left_y), (right_x, right_y),
                                     (int(color / 25 * q_idx), int(color / 25 * q_idx), int(color / 25 * q_idx)), 2)
            img_copy = cv2.circle(img_copy, (x, y), 1, display_color, -1)
            img_copy = cv2.putText(img_copy, str(q_idx), (left_x + 50, left_y + 50), \ \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 1)

            tube_mean_bright[i, j] = mean_bright
            print('{} REF BLOB SHAPE : {} \t\t ==> MEAN BRIGHT : {}'.format(q_idx, roi.shape, mean_bright))

            q_idx += 1

    plt.figure(figsize=(15, 15));
    plt.imshow(img_copy, cmap='gray', vmin=0, vmax=255 * 4), plt.title('TOTAL ROI'), plt.show();

    print('└━━━ Return func, [calc_blob_MeanBright] ━━━┘\n')

    return tube_mean_bright


# input img를 가로세로 grid(default=5)로 나누어 bright check
# q_point = 각 25개 blob point정보 len =26개 ()
def calc_grid_MeanBright(img, grid=5):
    print('\n┌━━━ Call func, [calc_grid_MeanBright] ━━━-┐')

    tube_mean_bright = np.zeros([5, 5])  # 각 tube mean bright 정보
    h, w = tube_mean_bright.shape[0:2]
    q_idx = 1

    # 해당부분 확인을 위한 그리기 img
    img_copy = np.copy(img)

    # grid 등분 h,w 구하기
    img_h, img_w = img_copy.shape
    grid_one_h = int(img_h / grid)
    grid_one_w = int(img_w / grid)

    print('IMG SHAPE : (h : {}, w : {})'.format(img_h, img_w))

    # 표시 color // img 구성 pixel의 90%
    color = int(np.unique(img)[int(len(np.unique(img)) * 0.9)])
    display_color = (color, color, color)

    for i in range(h):  # 5
        for j in range(w):  # 5

            left_x, left_y = grid_one_w * j, grid_one_h * i
            right_x, right_y = grid_one_w * (j + 1), grid_one_h * (i + 1)

            # 평균밝기 연산
            roi = img[left_y: right_y, left_x: right_x]  # 밝기 참조영역
            mean_bright = np.mean(roi)  # 평균밝기

            # 확인
            # plt.subplot(5,5,q_idx), plt.imshow(roi, cmap='gray', vmin=0, vmax=255), plt.title(str(q_idx) + ' of ROI') # plt로 그리기
            # plt.subplot(5,5,q_idx), sns.heatmap(roi, annot=True, fmt=".2f", cmap='gray', vmin = 0, vmax = 255*4), plt.title(str(q_idx) + ' of ROI') # sns으로 heatmap 그리기

            img_copy = cv2.rectangle(img_copy, (left_x, left_y), (right_x, right_y),
                                     (int(color / 25 * q_idx), int(color / 25 * q_idx), int(color / 25 * q_idx)),
                                     2)  # 그라데이션 color
            img_copy = cv2.putText(img_copy, str(q_idx), (left_x + 50, left_y + 50), \ \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 1)

            tube_mean_bright[i, j] = mean_bright
            print('{} GRID SHAPE : {} \t\t ==> MEAN BRIGHT : {}'.format(q_idx, roi.shape, mean_bright))

            q_idx += 1

    plt.figure(figsize=(15, 15));
    plt.imshow(img_copy, cmap='gray', vmin=0, vmax=255), plt.title('TOTAL ROI'), plt.show();  # vmax = 4*255

    print('└━━━ Return func, [calc_grid_MeanBright] ━━━┘\n')

    return tube_mean_bright