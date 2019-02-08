import numpy as np
import matplotlib.pyplot as plt
import math as mt
I = plt.imread('chessboard.png')
np.seterr(divide='ignore', invalid='ignore')
plt.imshow(I,cmap=plt.cm.gray)
plt.show()


def convolve(g,h, I_gray_copy): # h is kernel, g is the image
    start = 1
    for i in range(start,len(g[:,1])-1):
        for j in range(start, len(g[i,:])-1):
            f = g[i-1:i+2, j-1:j+2] #FIXME
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


I_u = plt.imread('chessboard.png')
I_v = plt.imread('chessboard.png')
s_h = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
s_ht = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

I_u = convolve(I, s_h, I_u)
I_v = convolve(I, s_ht, I_v)


def harris_response(I_u, I_v, I):
    g_kernel = gauss_kernal(3,2)
    I_uu = convolve(I_u*I_u, g_kernel, I_u)
    I_vv = convolve(I_v*I_v, g_kernel, I_v)
    I_uv = convolve(I_u*I_v, g_kernel, I)
    
    H = ((I_uu * I_uv) - (I_uv * I_uv))/ (I_uu + I_vv)
    
    return H

H = harris_response(I_u, I_v, I)
plt.imshow(H, cmap=plt.cm.gray)
plt.show()
