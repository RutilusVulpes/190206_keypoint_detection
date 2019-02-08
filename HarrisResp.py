import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.misc
import math as mt
import imageio as im

def convolve(g,h): # h is kernel, g is the image
    start = 
    x,y = h.shape()
    for i in range(start,len(g[:,1])-1):
        for j in range(start, len(g[i,:])-1):
            f = g[i-1:i+(x-1), j-1:j+(y-1)] #FIXME
            total = h*f
            I_gray_copy[i][j] = sum(sum(total)) 
    return I_gray_copy
           
def gauss_kernal(size, var):
    kernel = np.zeros(shape=(size,size))
    for i in range(size):
        for j in range(size):
            kernel[i][j] = mt.exp( -((i - (size-1)/2)**2 + (j - (size-1)/2)**2 )/(2*var*var))

    kernel = kernel / kernel.sum()
    return kernel           

def harris_response(img):
	sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
	gauss = gauss_kernal(5,2)
#calculate the harris response using sobel operator and gaussian kernel

	Iu = convolve(img,sobel)
	Iv = convolve(img,sobel.Transpose())

	Iuu = convolve(gauss,(Iu*Iu))
	Ivv = convolve(gauss,(Iv*Iv))
	Iuv = convolve(gauss,(Iu*Iv))

	H = (Iuv*Ivv - Iuv*Iuv)/(Iuu + Ivv)

	return H

