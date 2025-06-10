# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom

def T(dset):

    # 이미지 크기 축소 (픽셀 크기 유지) 변환 함수
    # 메타모픽 관계: 숫자 이미지를 축소해도 분류 결과는 변하지 않아야 함
    new_scale_dset = []
    for d in dset:
        # 이미지 중심을 기준으로 축소
        original_shape = d.shape

        scale_factor = 0.8  # 축소 비율
        
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

    # 노이즈 추가 변환 함수
    # 메타모픽 관계: 숫자 이미지에 적은 양의 노이즈를 추가해도 분류 결과는 변하지 않아야 함
    
    new_noise_dset = []
    noise_ratio = 0.05      # 노이즈를 추가할 픽셀의 비율
    noise_intensity = 0.3   # 노이즈의 강도
    
    for d in new_flip_dset:
        new_noise_d = d.copy()
        
        # MNIST 이미지 (height, width, channel) 형태로 노이즈 추가
        spatial_pixels = d.shape[0] * d.shape[1]  # height * width
        num_noise_pixels = int(spatial_pixels * noise_ratio)
        
        flat_indices = np.random.choice(spatial_pixels, num_noise_pixels, replace=False)
        rows, cols = np.unravel_index(flat_indices, (d.shape[0], d.shape[1]))
        
        noise_values = np.random.uniform(-noise_intensity, noise_intensity, num_noise_pixels)
        # 모든 채널에 동일한 노이즈 적용
        for c in range(d.shape[2]):
            new_noise_d[rows, cols, c] = np.clip(new_noise_d[rows, cols, c] + noise_values, 0, 1)
        
        new_noise_dset.append(new_noise_d)

    # =============================================================================================

    # 회전 변환 함수
    # 메타모픽 관계: 숫자 이미지를 약간 회전시켜도 분류 결과는 변하지 않아야 함

    new_rotate_dset = []
    rotate = 17 # 17도 단위로 회전 -> 테스트 20번 -> 20*17 = 340도 -> 18도로 할시 20번 테스트하면 360도로 마지막 한장이 원본 이미지와 같아짐
    for d in new_noise_dset:
        new_rotate_d = ndimage.rotate(d, rotate, reshape=False)
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
