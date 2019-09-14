import sys
import cv2
import numpy as np
import scipy.stats as st
import os
import math
import glob
import random


def calculate_parameters(points3D, points2D):

    A = np.zeros((len(points3D)*2, 12))

    j = 0
    for i in range(0, len(points3D)):
        x = np.array([[points3D[i][0], points3D[i][1], points3D[i][2], points3D[i][3], 0, 0, 0, 0, -points2D[i][0]*points3D[i][0], -points2D[i][0]*points3D[i][1], -points2D[i][0]*points3D[i][2], -points2D[i][0]*points3D[i][3]]])
        y = np.array([[0, 0, 0, 0, points3D[i][0], points3D[i][1], points3D[i][2], points3D[i][3], -points2D[i][1]*points3D[i][0], -points2D[i][1]*points3D[i][1], -points2D[i][1]*points3D[i][2], -points2D[i][1]*points3D[i][3]]])
        A[j] = x
        A[j+1] = y
        j+= 2

    ATA = np.dot(A.T, A)
    U, S, V = np.linalg.svd(ATA, full_matrices=True)
    sol = V[:,11]

    a1 = np.array([sol[0], sol[1], sol[2]])
    a2 = np.array([sol[4], sol[5], sol[6]])
    a3 = np.array([sol[8], sol[9], sol[10]])
    b = np.matrix([[sol[3]], [sol[7]], [sol[11]]])

    ro = 1/np.linalg.norm(a3)
    u0 = ro**2*(np.dot(a1, a3))
    v0 = ro**2*(np.dot(a2, a3))
    alfav = math.sqrt(ro**2*np.dot(a2,a2)-v0**2)
    s = (ro**4/alfav)*(np.dot(np.cross(a1, a3), np.cross(a2, a3)))
    alfau = math.sqrt(ro**2*np.dot(a1, a1)-s**2-u0**2)
    print("(u0,v0): ",u0,v0)
    print()
	
    print("(alfau,alfav): ",alfau,alfav)
    print()
	
    print("s: ",s)
    print()
	
    K = np.matrix([[alfau, s, u0],[0.0, alfav, v0],[0.0, 0.0, 1]])

    Kinverse = np.linalg.inv(K)
    r1 = np.cross(a2,a3)/np.linalg.norm(np.cross(a2,a3))
    r3 = a3
    r2 = np.cross(r3, r1)
    eps = np.sign(b[2][0])
    T = (np.dot(Kinverse,b))
    T *= eps*ro
    extrinsic_params = np.matrix([[r1[0], r1[1], r1[2], T[0][0]],[r2[0], r2[1], r2[2], T[1][0]],[r3[0], r3[1], r3[2], T[2][0]]])

	
    print("extrinsic_params", extrinsic_params)
    print("intrinsic_params", K)

    M = np.dot(K, extrinsic_params)

    return M

def calculate_MSE(points3D, points2D, M):

    points2DH = []
    points2D = []
    points2D_ref = []

    MSEx = 0
    MSEy = 0

    f = open("3Dpoints.txt", "r")
    for line in f:
        pointsDH3 = np.array([[float(line.split()[0]), float(line.split()[1]), float(line.split()[2]), 1.0]])
        a = (M*pointsDH3.T)
        points2DH.append(a)
    f.close()

    g = open("2Dpoints.txt", "r")
    for line in g:
        points2D_ref.append([float(line.split()[0]), float(line.split()[1])])
    g.close()

    np.seterr(divide='ignore', invalid='ignore')
    for point in points2DH:
        if(point[2]!=0):
            point[0] = point[0]/point[2]
            point[1] = point[1]/point[2]
            point[2] = point[2]/point[2]
            point = np.delete(point, (2), axis = 0)
            points2D.append(point)

    for i in range(0, len(points2D)):
        
        MSEx += (points2D_ref[i][0]-points2D[i][0])**2
        MSEy += (points2D_ref[i][1]-points2D[i][1])**2
    
    MSEx = MSEx/len(points2D_ref)
    MSEy = MSEy/len(points2D_ref)
    MSE = MSEx+MSEy

    return MSE





def main():
    f = open("3Dpoints.txt", "r")	
    g = open("2Dpoints.txt", "r")
    points3D = []
    points2D = []

    for line in f:
        points3D.append([float(line.split()[0]), float(line.split()[1]), float(line.split()[2]), 1.0])
    f.close()

    for line in g:
        points2D.append([float(line.split()[0]), float(line.split()[1])])
    g.close()
	
    M = calculate_parameters(points3D, points2D)
    MSE = calculate_MSE(points3D, points2D, M)
    

    print("M", M)
    print("MSE", MSE)
    
if __name__ == '__main__':
    main()
