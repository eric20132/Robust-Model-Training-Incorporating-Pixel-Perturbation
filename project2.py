# -*- coding: utf-8 -*-

import numpy as np
#import scipy
import matplotlib.pyplot as plt
import cvxpy as cp
import csv
#import imageio
from numpy import linalg as la
#from scipy.stats import norm

def cov_mat(data, mean, d, m):
    cov = np.matrix(np.zeros((d,d)))
    for i in range(m):
        x = data.T[i]- mean.T
        cov = cov + x.T*x
    return cov/(m-1)

def dsc_fun(x, mean, cov_inv, res):
    y = x-mean
    return -0.5*(y.T*cov_inv*y)+res

def cat_grass_overlapping(Y):
    (M, N) = Y.shape
    output = np.zeros((M-8,N-8))
    for i in range(M-8):
        for j in range(N-8):
            z = Y[i:i+8, j:j+8]
            z_flt = z.flatten('F').reshape((d,1))
            g_cat = dsc_fun(z_flt, mean_cat, inv_cat, res_cat)
            g_grass = dsc_fun(z_flt, mean_grass, inv_grass, res_grass)
            if (g_cat >= g_grass):
                output[i, j] = 1
    return output

def cat_grass_nonoverlapping(Y):
    (M, N) = Y.shape
    output = np.zeros((M,N))
    for i in range(0, M-8, 8):
        for j in range(0, N-8, 8):
            z = Y[i:i+8, j:j+8]
            z_flt = z.flatten('F').reshape((d,1))
            g_cat = dsc_fun(z_flt, mean_cat, inv_cat, res_cat)
            g_grass = dsc_fun(z_flt, mean_grass, inv_grass, res_grass)
            if (g_cat >= g_grass):
                output[i:i+8, j:j+8] = np.ones((8,8))
    return output

def gradient_nonoverlapping(pert_vec, img_vec, target_index, lam):
    # Calculate g_j, g_t, determine if patch_vec is already in target class
    grad = 0
    tc = 1-2*target_index
    tg = 2*target_index-1
    g_cat = dsc_fun(pert_vec, mean_cat, inv_cat, res_cat)
    g_grass = dsc_fun(pert_vec, mean_grass, inv_grass, res_grass)
    # If patch_vec is in target class, do not add any perturbation (return zero gradient!)
    # Else, calculate the gradient, using results from 1(c)(ii)
    if (tc*g_cat+tg*g_grass > 0):
        grad = 2*(pert_vec-img_vec)-lam*(tc*inv_cat*(pert_vec-mean_cat)+tg*inv_grass*(pert_vec-mean_grass))
    return grad

def gradient_overlapping(pert_vec, target_index):
    # Calculate g_j, g_t, determine if patch_vec is already in target class
    grad = np.zeros((d,1))
    tc = 1-2*target_index
    tg = 2*target_index-1
    g_cat = dsc_fun(pert_vec, mean_cat, inv_cat, res_cat)
    g_grass = dsc_fun(pert_vec, mean_grass, inv_grass, res_grass)
    # If patch_vec is in target class, do not add any perturbation (return zero gradient!)
    # Else, calculate the gradient, using results from 1(c)(ii)
    if (tc*g_cat+tg*g_grass > 0):
        grad = -tc*inv_cat*(pert_vec-mean_cat)-tg*inv_grass*(pert_vec-mean_grass)
    return grad

def CW_attack_nonoverlapping(img_matrix, target_index, alpha, lam):
    # Preprocess the data, and set up parameters
    (M, N) = img_matrix.shape
    pert_img_prev = np.zeros((M, N))
    pert_img_curr = img_matrix
    change = np.linalg.norm(pert_img_curr - pert_img_prev)
    itr_num = 0
    pert_img_1 = np.zeros((M, N))
    pert_img_2 = np.zeros((M, N))
    pert_img_3 = np.zeros((M, N))
    # Start gradient descent
    while itr_num <= 300 and change >= 0.001:
        pert_img_prev = pert_img_curr.copy()
        for i in range(M):
            for j in range(N):
                if i + 8 > M or j + 8 > N or i % 8 != 0 or j % 8 != 0:
                    continue
                # Perturb the patch with top left pixel (i,j) in img_matrix by
                x = img_matrix[i:i+8, j:j+8]
                z = pert_img_prev[i:i+8, j:j+8]
                img_vec = x.flatten('F').reshape((d,1))
                pert_vec = z.flatten('F').reshape((d,1))
                grad = gradient_nonoverlapping(pert_vec, img_vec, target_index, lam)
                pert_vec = np.clip(pert_vec - alpha*grad, 0.0, 1.0)
                pert_img_curr[i:i+8, j:j+8] = np.reshape(pert_vec, (8, 8), order = 'F')                
        # Process the perturbed image matrix, display perturbed image, etc.
        change = np.linalg.norm(pert_img_curr - pert_img_prev)
        itr_num += 1
        if itr_num == int(50/lam):
            pert_img_1 = pert_img_curr.copy()
        elif itr_num == int(100/lam): 
            pert_img_2 = pert_img_curr.copy()
        elif itr_num == int(150/lam): 
            pert_img_3 = pert_img_curr.copy()  
    # Return the attack image
    print(itr_num)
    return pert_img_1, pert_img_2, pert_img_3, pert_img_curr

def CW_attack_overlapping(img_matrix, target_index, alpha, lam):
    # Preprocess the data, and set up parameters
    (M, N) = img_matrix.shape
    pert_img_prev = np.zeros((M, N))
    pert_img_curr = img_matrix
    change = np.linalg.norm(pert_img_curr - pert_img_prev)
    itr_num = 0
    clist = []
    pert_img_1 = np.zeros((M, N))
    pert_img_2 = np.zeros((M, N))
    pert_img_3 = np.zeros((M, N))
    pert_img_4 = np.zeros((M, N))
    pert_img_5 = np.zeros((M, N))
    pert_img_6 = np.zeros((M, N))
    pert_img_7 = np.zeros((M, N))
    # Start gradient descent
    while itr_num <= 60 and change >= 0.01:
        pert_img_prev = pert_img_curr.copy()
        pert_img_curr = pert_img_curr-2*alpha*(pert_img_prev-img_matrix)
        for i in range(M-8):
            for j in range(N-8):
                # Perturb the patch with top left pixel (i,j) in img_matrix by
                # pert_patch = patch_vec + alpha*grad
                z = pert_img_prev[i:i+8, j:j+8]
                pert_vec = z.flatten('F').reshape((d,1))
                grad = gradient_overlapping(pert_vec, target_index)
                pert_img_curr[i:i+8, j:j+8] += np.reshape(-alpha*lam*grad, (8, 8), order = 'F')                
        # Process the perturbed image matrix, display perturbed image, etc.
        pert_img_curr = np.clip(pert_img_curr, 0.0, 1.0)
        # change = np.linalg.norm(pert_img_curr - pert_img_prev)
        itr_num += 1
        print(itr_num)
        print(change)
        # Classification
        clist.append(cat_grass_nonoverlapping(pert_img_curr.copy()))

        if itr_num == 5:
            pert_img_1 = pert_img_curr.copy()
        elif itr_num == 10: 
            pert_img_2 = pert_img_curr.copy()
        elif itr_num == 15: 
            pert_img_3 = pert_img_curr.copy() 
        elif itr_num == 20: 
            pert_img_4 = pert_img_curr.copy() 
        elif itr_num == 25: 
            pert_img_5 = pert_img_curr.copy() 
        elif itr_num == 30: 
            pert_img_6 = pert_img_curr.copy() 
        elif itr_num == 35: 
            pert_img_7 = pert_img_curr.copy() 
    # Return the attack image
    return pert_img_1, pert_img_2, pert_img_3, pert_img_4, pert_img_5, pert_img_6, pert_img_7, pert_img_curr, clist

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def accuracy_cal_check(cat_list,truth):
    accu_list = []
    for i in range(len(cat_list)):
        accu_list.append(np.sum(np.equal(cat_list[i] ,truth))/(33135))
    return accu_list


# train_cat = np.matrix(np.loadtxt('data/train_cat.txt', delimiter = ','))
# train_cat = np.matrix(np.loadtxt('data/train_cat_noise.txt', delimiter = ','))
train_cat = np.matrix(np.loadtxt('data/train_cat_flipnoise.txt', delimiter = ','))

# train_grass = np.matrix(np.loadtxt('data/train_grass.txt', delimiter = ','))
# train_grass = np.matrix(np.loadtxt('data/train_grass_noise.txt', delimiter = ','))
train_grass = np.matrix(np.loadtxt('data/train_grass_flipnoise.txt', delimiter = ','))

(d, num_cat) = train_cat.shape
(d, num_grass) = train_grass.shape    

mean_cat = train_cat.mean(1)
mean_grass = train_grass.mean(1)
cov_cat = cov_mat(train_cat, mean_cat, d, num_cat)
cov_grass = cov_mat(train_grass, mean_grass, d, num_grass)
prior_cat = num_cat/(num_cat+num_grass)
prior_grass = num_grass/(num_cat+num_grass)

det_cat = la.det(cov_cat)
det_grass = la.det(cov_grass)
inv_cat = cov_cat.I
inv_grass = cov_grass.I
res_cat = -32*np.log(2*np.pi) - 0.5*np.log(det_cat) + np.log(prior_cat)
res_grass = -32*np.log(2*np.pi) - 0.5*np.log(det_grass) + np.log(prior_grass)
img = plt.imread('data/cat_grass.jpg')/255


# alpha = 0.0001
# lam = 10
"""
pert_img = CW_attack_nonoverlapping(img.copy(), 0, alpha, lam)
plt.figure()
plt.imshow((pert_img)*255, cmap = 'gray')
plt.savefig("2ci6.pdf")
pert_add = pert_img-img
plt.figure()
plt.imshow((pert_add)*255, cmap = 'gray')
plt.savefig("2cii6.pdf")
print (np.linalg.norm(pert_add))
output1 = cat_grass_nonoverlapping(pert_img)
plt.figure()
plt.imshow(output1*255, cmap = 'gray')
plt.savefig("2civ6.pdf")
"""
"""
pert_img_1, pert_img_2, pert_img_3, pert_img = CW_attack_nonoverlapping(img.copy(), 0, alpha, lam)
output1 = cat_grass_nonoverlapping(pert_img_1)
output2 = cat_grass_nonoverlapping(pert_img_2)
output3 = cat_grass_nonoverlapping(pert_img_3)
output = cat_grass_nonoverlapping(pert_img)
plt.figure()
plt.imshow(output1*255, cmap = 'gray')
plt.savefig("2cv6-1.pdf")
plt.figure()
plt.imshow(output2*255, cmap = 'gray')
plt.savefig("2cv6-2.pdf")
plt.figure()
plt.imshow(output3*255, cmap = 'gray')
plt.savefig("2cv6-3.pdf")
"""

truth = plt.imread('data/truth.png')
truth[truth < .5] = 2
alpha = 0.001
lam = 0.007
clist=[]
accuList = []
pert_img_1, pert_img_2, pert_img_3, pert_img_4, pert_img_5, pert_img_6, pert_img_7, pert_img, clist = CW_attack_overlapping(img.copy(), 0, alpha, lam)
accuList = accuracy_cal_check(clist,truth)

with open('accuracy/accuracy3.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
     wr.writerow(accuList)

# output1 = cat_grass_overlapping(pert_img_1)
# output2 = cat_grass_overlapping(pert_img_2)
# output3 = cat_grass_overlapping(pert_img_3)
# output4 = cat_grass_overlapping(pert_img_4)
# output5 = cat_grass_overlapping(pert_img_5)
# output6 = cat_grass_overlapping(pert_img_6)
# output7 = cat_grass_overlapping(pert_img_7)
# output = cat_grass_overlapping(pert_img)
# pert_add = pert_img-img
# print (np.linalg.norm(pert_add))

# plt.figure()
# plt.imshow((pert_img)*255, cmap = 'gray')
# plt.savefig("3c4-p.pdf")
# plt.figure()
# plt.imshow((pert_add)*255, cmap = 'gray')
# plt.savefig("3c4-a.pdf")
# plt.figure()
# plt.imshow(output*255, cmap = 'gray')
# plt.savefig("3c4-o.pdf")
# plt.figure()
# plt.imshow(output1*255, cmap = 'gray')
# plt.savefig("3c4-1.pdf")
# plt.figure()
# plt.imshow(output2*255, cmap = 'gray')
# plt.savefig("3c4-2.pdf")
# plt.figure()
# plt.imshow(output3*255, cmap = 'gray')
# plt.savefig("3c4-3.pdf")
# plt.figure()
# plt.imshow(output4*255, cmap = 'gray')
# plt.savefig("3c4-4.pdf")
# plt.figure()
# plt.imshow(output5*255, cmap = 'gray')
# plt.savefig("3c4-5.pdf")
# plt.figure()
# plt.imshow(output6*255, cmap = 'gray')
# plt.savefig("3c4-6.pdf")
# plt.figure()
# plt.imshow(output7*255, cmap = 'gray')
# plt.savefig("3c4-7.pdf")
