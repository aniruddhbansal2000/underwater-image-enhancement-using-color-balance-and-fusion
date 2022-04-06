import cv2
import numpy as np

# function to calculate Laplacian contrast weight
def laplacian_weight(I):
  # applying laplacian filter for all the 3 channels
  LapR = cv2.Laplacian(I[:, :, 0],cv2.CV_64F)
  LapG = cv2.Laplacian(I[:, :, 1],cv2.CV_64F)
  LapB = cv2.Laplacian(I[:, :, 2],cv2.CV_64F)

  # mean value for the laplacian filter
  LapR_mean = LapR.mean()
  LapG_mean = LapG.mean()
  LapB_mean = LapB.mean()

  laplacian_mat = ((LapR**2 + LapG**2 + LapB**2)/3) ** 0.5
  # laplacian_mat = (((I[:, :, 0] - LapR_mean)**2 + (I[:, :, 1] - LapG_mean)**2 + (I[:, :, 2] - LapB_mean)**2)/3) ** 0.5 # another global contrast weight using laplacian filter 
  
  return laplacian_mat

# Saliency weight calculation
def saliency_weight(I_lab):
  
  L = I_lab[:, :, 0]
  A = I_lab[:, :, 1]
  B = I_lab[:, :, 2]

  L_mean = np.mean(L)
  A_mean = np.mean(A)
  B_mean = np.mean(B)

  saliency_mat = ((L - L_mean) ** 2) + ((A - A_mean) ** 2) + ((B - B_mean) ** 2)
  return (saliency_mat/ saliency_mat.max())

def saturation_weight(I, R1):

  R = I[:, :, 0]
  G = I[:, :, 1]
  B = I[:, :, 2]

  saturation_mat = (((R - R1)**2 + (G - R1)**2 + (B - R1)**2)/3) ** 0.5
  return saturation_mat

