import numpy as np
import matplotlib.pyplot as plt
I = plt.imread('chessboard.png')
I2 = plt.imread('chessboard.png')
I3 = plt.imread('chessboard.png')
I4 = plt.imread('chessboard.png')
I5 = plt.imread('chessboard.png')
I6 = plt.imread('chessboard.png')
plt.imshow(I,cmap=plt.cm.gray)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.misc
import math as mt
import imageio as im

def convolve(g,h, I_gray_copy): # h is kernel, g is the image
   
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

def harris_response(img, gmean = 5,var =2):
	sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
	gauss = gauss_kernal(gmean,var)
#calculate the harris response using sobel operator and gaussian kernel

	Iu = convolve(img,sobel,I2)
	Iv = convolve(img,sobel.transpose(), I3)

	Iuu = convolve(Iu*Iu,gauss, I4)
	Ivv = convolve(Iv*Iv,gauss, I5)
	Iuv = convolve(Iu*Iv,gauss, I6)



	H = (Iuv*Ivv - Iuv*Iuv)/(Iuu + Ivv)

	return H

H = harris_response(I,gmean = 5,var=2)


plt.imshow(H,cmap=plt.cm.gray)
#.imshow(I,cmap=plt.cm.gray)
plt.show()
print(H)

def get_local_maxima(H):
    localMax = []
    
    for i in range(1,len(H)-1):
        for j in range(1,len(H[i])-1):
            neighborhood = H[i-1:i+2, j-1:j+2]
            max_val = max(max(x) for x in neighborhood)
            if(max_val == H[i][j] and max_val!= 0):
                localMax.append([i,j,H[i][j],np.inf])
    return localMax



local_max = get_local_maxima(H)


c = .9
def distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

num_keypts = len(local_max)

x = 0

while(x < num_keypts):
    for j in range(num_keypts):
        if(j != x):
            if(local_max[j][2]*c > local_max[x][2]):
                if(distance(local_max[j][0],local_max[j][1], local_max[x][0], local_max[x][1]) < local_max[x][3]):
                    local_max[x][3] = distance(local_max[j][0],local_max[j][1], local_max[x][0], local_max[x][1])
    x+=1

print("Finished")

def sort_dist(val):
    return val[3]

local_max.sort(reverse=True, key=sort_dist)

print(local_max[0][3])
