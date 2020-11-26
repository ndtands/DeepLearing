from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt

# Diện tích (m^2)
X = np.array([[30,32.4138,34.8276,37.2414,39.6552,41.523,45.314,49.1243,55.124,59.123,62.113,67.153,70.342]]).T
# giá tiền (triệu)
y = np.array([[448.524,509.248,535.104,551.432,623.418,628.133,687.512,744.123,833.125,900.22,935.164,1012.342,1064.13]]).T
# Visualize data 
plt.plot(X, y, 'ro')
plt.axis([25, 72, 400, 1100])
plt.xlabel('Dien tich (m^2)')
plt.ylabel('Gia tien (trieu)')
plt.show()

# Building Xbar 
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

# Lưu w với numpy.save(), định dạng '.npy'
np.save('weight.npy', w)
# Đọc file '.npy' chứa tham số weight
w = np.load('weight.npy')

print('w = ', w)
# Preparing the fitting line 
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(25,73, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line 
plt.plot(X.T, y.T, 'ro')     # data 
plt.plot(x0, y0)               # the fitting line
plt.axis([25, 73, 400, 1100])
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show()

#predict
y1 = w_1*50 + w_0
y2 = w_1*100 + w_0

print( u'Predict price with height 50 m^2: %.2f (Trieu)'  %(y1) )
print( u'Predict price with height 60 m^2: %.2f (Trieu)'  %(y2) )