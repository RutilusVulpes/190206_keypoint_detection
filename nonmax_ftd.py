#  Non-Maximal Suppression Rough Draft

import numpy as np

def getmaxima (H,threshold):
    maxima = []
    x,y = shape(H)
    for i in range(1,x-1):
        for j in range(1,y-1):
            if H[i,j] > threshold :
                continue
            
            if(    H[i,j] > H[i+1,j]   and  H[i,j] > H[i-1,j] 
               and H[i,j] > H[i,j-1]   and  H[i,j] > H[i,j+1]
               and H[i,j] > H[i+1,j-1] and  H[i,j] > H[i+1,j+1]
               and H[i,j] > H[i-1,j-1] and  H[i,j] > H[i-1,j+1]):
                
                maxima.append(i,j,H[i,j])
    return maxima 
    
    
def nonmaxsup(H,n=100,c=.9):
    
    mindistance = []
    threshold = np.mean(H) + np.stddev(H)
    maxima = getmaxima(H,threshold)
    
    for x,y,z in enumerate(maxima):
        min = np.infinity
        for xx,yy,zz in enumerate(maxima):
            dist = sqrt((x-xx)**2 + (y-yy)**2 )
            if z < c*zz and dist > 0 and dist < min:
                min = dist
                xmin = xx
                ymin = yy

        mindistance.append((xx,yy,min))

    mindistance.sort(key=lambda x:x[2])
    return mindistance[:n]



		