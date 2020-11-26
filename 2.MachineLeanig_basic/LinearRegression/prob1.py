'''
Dùng giải thuật gradient descent để tìm giá trị nhỏ nhất của hàm số
    f(x) = x^2 + 2x +5
'''
''' Gradient Descent with singer parameter
    Function: f(x)= x^2 + 2*x+5
    => f'(x) = 2x+ 2
    x_t+1 = x_t-n*f'(x_t)
    n: Learning rate
'''
import math
import numpy as np 
import matplotlib.pyplot as plt

def grad(x):
    return 2*x+ 2

def cost(x):
    return x**2 + 2*x+5
#function using để mô tả quá trình tiến đến vị trí mà đạo hầm tại đó bằng 0
def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1]) # x_t+1 = x_t - n*f(x_t)'
        if abs(grad(x_new)) < 1e-3: #err <10^-3
            break
        x.append(x_new)
    return (x, it)

#innit with x0 = -5
(x1, it1) = myGD1(.1, -5)
#innit with x0 = 5
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))



#innit with x0 = 5 and n=0.01
(x1, it1) = myGD1(.01,5)
#innit with x0 = 5 and n= 0.07
(x2, it2) = myGD1(.07, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))
