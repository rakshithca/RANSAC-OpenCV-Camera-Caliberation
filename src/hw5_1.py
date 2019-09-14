import cv2
import numpy as np
import glob


def main():

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    co_or = np.zeros((6*7,3), np.float32)
    co_or[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    pt3D = []
    pt2D = []

    imgs = glob.glob('*.jpg')

    for filename in imgs:
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img_gray,(7,6),None)

        if ret == True:
            pt3D.append(co_or)
            corners_new = cv2.cornerSubPix(img_gray,corners,(11,11),(-1,-1),criteria)
            pt2D.append(corners_new)

            img = cv2.drawChessboardCorners(img, (7,6), corners_new,ret)
            cv2.imshow('img',img)

    
    f = open("2Dpoints.txt","w")
    for point in pt2D[0]:
        for p in point[0]:
            f.write("%f " %p)
        f.write("\n")
    f.close()

    g = open("3Dpoints.txt","w")
    for point in pt3D[0]:
        for p in point:
            g.write("\t%f" %p)
        g.write("\n")
    g.close()

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
