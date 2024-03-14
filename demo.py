import math 

x_0 = 2
x_1 = 1.8

for i in range(0, 8):
    x_n = 1
    x_n1 = 2
    x_1 = 1.8
    x_n1 = x_n - (x_n ** 3 - 3 * x_n - 1) / (x_n ** 3 - 3 * x_n - x_n1 ** 3 + 3 * x_n1) * (x_n - x_1) 
