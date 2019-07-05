import numpy as np
import matplotlib.pyplot as plt
#input-number means x distribution ,we use bernulli distribution so we have only two in put ,and input number 2 means
# p=0.5 input number 3 means p=o.33333 input number 4 means p= 0.25

def H2(x):
    x=1/x
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def HH(D):
    return -D * np.log2(D) - (1 - D) * np.log2(1 - D)

RR =np.zeros((50,1) ,float)
DD =np.zeros((50,1) ,float)

input_num=2
D = 0
c = 0
while c < 50:
    if D == 0:
        RD = H2(input_num)
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
    else:
        RD = H2(input_num) - HH(D)
        print(H2(input_num),HH(D))
        if RD <= 0:
            RD = 0
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
plt.plot(DD, RR)


input_num=3
D = 0
c = 0
while c < 50:
    if D == 0:
        RD = H2(input_num)
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
    else:
        RD = H2(input_num) - HH(D)
        print(H2(input_num),HH(D))
        if RD <= 0:
            RD = 0
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
    print('D==',D)
plt.plot(DD, RR)

input_num=4
D = 0
c = 0
while c < 50:
    if D == 0:
        RD = H2(input_num)
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
    else:
        RD = H2(input_num) - HH(D)
        print(H2(input_num),HH(D))
        if RD <= 0:
            RD = 0
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
plt.plot(DD, RR)

input_num=5
D = 0
c = 0
while c < 50:
    if D == 0:
        RD = H2(input_num)
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
    else:
        RD = H2(input_num) - HH(D)
        print(H2(input_num),HH(D))
        if RD <= 0:
            RD = 0
        RR[c] = RD
        DD[c] = D
        c += 1
        D += 0.01
plt.plot(DD, RR)

# input_num=6
# D = 0
# c = 0
# while c < 50:
#     if D == 0:
#         RD = H2(input_num)
#         RR[c] = RD
#         DD[c] = D
#         c += 1
#         D += 0.01
#     else:
#         RD = H2(input_num) - HH(D)
#         print(H2(input_num),HH(D))
#         if RD <= 0:
#             RD = 0
#         RR[c] = RD
#         DD[c] = D
#         c += 1
#         D += 0.01
# plt.plot(DD, RR,'c')


plt.show()