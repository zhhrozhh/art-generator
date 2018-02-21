import numpy as np

def gauss_denoise(X,ksize = 5):
	assert ksize%2
	hwin = (ksize-1)//2
	w,h,c = X.shape
	WX = np.pad(X,((hwin,hwin),(hwin,hwin),(0,0)))
	res = np.zeros((w,h,z))
	for i in range(0,w):
		for j in range(0,h):
			for c in range(3):
				res[i,j,c] = WX[i:i+ksize,j+j+ksize,c].mean()
	return res 