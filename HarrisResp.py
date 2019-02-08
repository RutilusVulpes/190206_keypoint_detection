
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.misc
import math as mt
import imageio as im

def convolve(g,h): # h is kernel, g is the image
    I_gray_copy = g
   
    x,y = h.shape
    xl = int(x/2)
    yl = int(y/2)
    for i in range(xl,len(g[:,1])-xl):
        for j in range(yl, len(g[i,:])-yl):

            f = g[i-xl:i+(xl+1), j-yl:j+(yl+1)] #FIXME
            
            total = h*f
            I_gray_copy[i][j] = sum(sum(total)) + .0000001 
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
	Iv = convolve(img,sobel.transpose())

	Iuu = convolve((Iu*Iu),gauss)
	Ivv = convolve((Iv*Iv),gauss)
	Iuv = convolve((Iu*Iv),gauss)

	H = (Iuv*Ivv - Iuv*Iuv)/(Iuu + Ivv)

	return H
