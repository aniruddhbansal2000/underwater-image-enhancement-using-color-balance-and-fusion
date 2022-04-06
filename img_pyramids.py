import scipy
import numpy as np
import cv2

def gaussian_pyramid(img, level):
  h = (1/16.0) * np.array([[1, 4, 6, 4, 1]])
  h_trans = np.transpose(h) 
  filt = np.dot(h_trans, h)

  out = []
  tmp_img = img
  for i in range(level):
    tmp_arr = scipy.ndimage.convolve(tmp_img, filt, mode='nearest')
    out.append(tmp_arr)
    tmp_img = tmp_img[::2, ::2]
  return out

def laplacian_pyramid(img, level):
  h = (1/16.0) * np.array([[1, 4, 6, 4, 1]])
  
  out = []
  tmp_img = img
  for i in range(level):
    out.append(tmp_img)
    tmp_img = tmp_img[::2, ::2]

  # calculate DoG
  for i in range(level-1):
    
    (timg_w, timg_h) = (out[i].shape[0], out[i].shape[1])
    tmp_resized = (cv2.resize(out[i + 1], (timg_w, timg_h))).T  # opencv returns transposed result on resizing, transpose it back!
 
    # modify out[i], as list, can't modify by index, remove/add-corrected
    new_out_i = out[i] - tmp_resized
    out.pop(i)
    out.insert(i, new_out_i)

  return out

def pyramid_reconstruct(pyramid):
  level = len(pyramid)
  for i in range(level-1, 0, -1):
    (timg_w, timg_h) = (pyramid[i - 1].shape[0], pyramid[i - 1].shape[1])
    tmp_resized = (cv2.resize(pyramid[i], (timg_w, timg_h))).T  # opencv returns transposed result on resizing, transpose it back!
 
    # modify pyramid[i - 1], as list, can't modify by index, remove/add-corrected
    new_pyramid_i_minus = pyramid[i - 1] + tmp_resized
    pyramid.pop(i - 1)
    pyramid.insert(i - 1, new_pyramid_i_minus)

  out = pyramid[0]
  return out
