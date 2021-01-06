import numpy as np
import matplotlib.pyplot as plt

filePoints = open('/home/lu/code/CLionProjects/CurveFitting/scripts/data.txt')
dataPoints = filePoints.readlines()

fileResult = open('/home/lu/code/CLionProjects/CurveFitting/scripts/result.txt')
dataResult = fileResult.readlines()

fileErrorGN = open('/home/lu/code/CLionProjects/CurveFitting/scripts/error-GN.txt')
dataErrorGN = fileErrorGN.readlines()

fileErrorRGN = open('/home/lu/code/CLionProjects/CurveFitting/scripts/error-RGN.txt')
dataErrorRGN = fileErrorRGN.readlines()

fileErrorLM = open('/home/lu/code/CLionProjects/CurveFitting/scripts/error-LM.txt')
dataErrorLM = fileErrorLM.readlines()

fileErrorLMN = open('/home/lu/code/CLionProjects/CurveFitting/scripts/error-LM-N.txt')
dataErrorLMN = fileErrorLMN.readlines()

fileErrorRLM = open('/home/lu/code/CLionProjects/CurveFitting/scripts/error-RLM.txt')
dataErrorRLM = fileErrorRLM.readlines()

fileErrorCERES = open('/home/lu/code/CLionProjects/CurveFitting/scripts/error-ceres.txt')
dataErrorCERES = fileErrorCERES.readlines()

points_x = []       # points
points_y = []
gn_iter = []        # GN
gn_error = []
rgn_iter = []       # RGN
rgn_error = []
lm_iter = []        # LM
lm_error = []
lmn_iter = []       # LM-Nielsen
lmn_error = []
rlm_iter = []       # RLM
rlm_error = []
ceres_iter = []     # CERES
ceres_error = []

for num in dataPoints:
    points_x.append(float(num.split(',')[0]))
    points_y.append(float(num.split(',')[1]))

for num in dataErrorGN:
    gn_iter.append(float(num.split(',')[0]))
    gn_error.append(float(num.split(',')[1]))

for num in dataErrorRGN:
    rgn_iter.append(float(num.split(',')[0]))
    rgn_error.append(float(num.split(',')[1]))

for num in dataErrorLM:
    lm_iter.append(float(num.split(',')[0]))
    lm_error.append(float(num.split(',')[1]))

for num in dataErrorLMN:
    lmn_iter.append(float(num.split(',')[0]))
    lmn_error.append(float(num.split(',')[1]))

for num in dataErrorRLM:
    rlm_iter.append(float(num.split(',')[0]))
    rlm_error.append(float(num.split(',')[1]))

for num in dataErrorCERES:
    ceres_iter.append(float(num.split(',')[0]))
    ceres_error.append(float(num.split(',')[1]))

x = np.linspace(0, 1, 1000)
ar, br, cr = 1.0, 2.0, 1.0
yr = np.exp(ar * x * x + br * x + cr)
ae, be, ce = 1.00209, 1.993, 1.00636
ye = np.exp(ae * x * x + be * x + ce)

# print(dataResult[0])

# figure
plt.figure(figsize=(12, 6))
# curve plot
plt.subplot(1, 2, 1)
plt.title('exp(ar * x * x + br * x + cr)')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(points_x, points_y, 0.1, c='b', alpha=0.8, label="points")
plt.plot(x, ye, linewidth=1, c='c', label='GN')
plt.plot(x, yr, linewidth=1, c='r', alpha=0.8, label='real')
plt.legend(loc='best')
# cost descent
plt.subplot(1, 2, 2)
plt.title('error')
plt.xlabel('iter')
plt.ylabel('error')

plt.plot(gn_iter, gn_error, linewidth=0.5, c='r')
plt.scatter(gn_iter, gn_error, s=5, c='r', marker='o', alpha=0.5, label='GN')
#
# plt.plot(lm_iter, lm_error, linewidth=0.5, c='b')
# plt.scatter(lm_iter, lm_error, s=5, c='b', marker='^', alpha=0.5, label='LM')
#
# plt.plot(lmn_iter, lmn_error, linewidth=0.5, c='g')
# plt.scatter(lmn_iter, lmn_error, s=5, c='g', marker='x', alpha=0.5, label='LMN')
#
# plt.plot(rgn_iter, rgn_error, linewidth=0.5, c='r')
# plt.scatter(rgn_iter, rgn_error, s=5, c='r', marker='o', alpha=0.5, label='RGN')

plt.plot(rlm_iter, rlm_error, linewidth=0.5, c='b')
plt.scatter(rlm_iter, rlm_error, s=5, c='b', marker='^', alpha=0.5, label='RLM')

plt.plot(ceres_iter, ceres_error, linewidth=0.5, c='g')
plt.scatter(ceres_iter, ceres_error, s=5, c='g', marker='x', alpha=0.5, label='CERES')

plt.legend(loc='best')
# show
plt.show()
