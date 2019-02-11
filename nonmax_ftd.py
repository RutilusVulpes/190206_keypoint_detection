#  Non-Maximal Suppression Rough Draft

import numpy as np

def nonmaxsup(H,n=100,c=.9):
	mindistance = []
	for x,y,z in enumerate(H):
		min = np.infinity
		for xx,yy,zz in enumerate(H):
			dist = sqrt((x-xx)*2 + (y-yy)*2 + (z - zz)*2)
			if z < c*zz and dist > 0 and dist < min:
				min = dist
				xmin = xx
				ymin = yy

		mindistance.append((xx,yy,min))

	mindistance.sort(key=lambda x:x[2])
	return mindistance[:n]



		