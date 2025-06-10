# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom

# 전역 회전 카운터 (번갈아가며 회전하기 위함)
_rotation_counter = 0

def T(dset):

    # 이미지 크기 축소 (픽셀 크기 유지) 변환 함수
    # 메타모픽 관계: 숫자 이미지를 축소해도 분류 결과는 변하지 않아야 함
    new_scale_dset = []
    for d in dset:
        # 이미지 중심을 기준으로 축소
        original_shape = d.shape

        scale_factor = 0.99  # 축소 비율
        
        # 확대/축소 수행
        scaled = zoom(d, scale_factor, order=1)
        
        # 원본 크기로 padding 추가
        scaled_shape = scaled.shape
        new_scale_d = np.zeros(original_shape)
        start_h = (original_shape[0] - scaled_shape[0]) // 2
        start_w = (original_shape[1] - scaled_shape[1]) // 2
        new_scale_d[start_h:start_h + scaled_shape[0], 
                start_w:start_w + scaled_shape[1]] = scaled
            
        new_scale_dset.append(new_scale_d)

    # =============================================================================================


    # 이미지 좌우반전 변환 함수
    # 메타모픽 관계: 숫자 이미지를 좌우반전해도 분류 결과는 변하지 않아야 함
    
    new_flip_dset = []
    for d in new_scale_dset:
        new_flip_d = np.fliplr(d)
        new_flip_dset.append(new_flip_d)

    # =============================================================================================

    # 픽셀 블록 반전 변환 함수
    # 메타모픽 관계: 숫자 이미지에 픽셀 블록을 반전해도 분류 결과는 변하지 않아야 함
    
    new_invert_dset = []
    block_size = 1     # 반전할 블록의 크기
    
    for d in new_flip_dset:
        new_invert_d = d.copy()
        
        # 이미지 크기에서 블록을 배치할 수 있는 범위 계산
        max_row = d.shape[0] - block_size
        max_col = d.shape[1] - block_size
        
        # 랜덤한 위치에 블록 배치
        if max_row > 0 and max_col > 0:
            start_row = np.random.randint(0, max_row + 1)
            start_col = np.random.randint(0, max_col + 1)
            
            # 10x10 영역의 픽셀 값을 반전 (검은색↔흰색)
            for c in range(d.shape[2]):
                block = new_invert_d[start_row:start_row + block_size, 
                                   start_col:start_col + block_size, c]
                # 픽셀 값 반전: 0 → 1, 1 → 0
                new_invert_d[start_row:start_row + block_size, 
                           start_col:start_col + block_size, c] = 1.0 - block
        
        new_invert_dset.append(new_invert_d)

    # =============================================================================================

    # 회전 변환 함수
    # 메타모픽 관계: 숫자 이미지를 약간 회전시켜도 분류 결과는 변하지 않아야 함

    global _rotation_counter
    new_rotate_dset = []
    
    # 회전 방향을 번갈아가며 적용 (17도, -17도)
    # 좌우반전 때문에 한 방향으로만 회전하면 패턴이 반복될 수 있음
    rotate_angle = 17 if _rotation_counter % 2 == 0 else -17
    _rotation_counter += 1
    
    for d in new_invert_dset:
        new_rotate_d = ndimage.rotate(d, rotate_angle, reshape=False)
        new_rotate_dset.append(new_rotate_d)

    return np.array(new_rotate_dset)

def E(source_y, follow_y):
    result = []
    for s, f in zip(source_y, follow_y):
        if s == f:
            result.append(True)
        else:
            result.append(False)
    return result
