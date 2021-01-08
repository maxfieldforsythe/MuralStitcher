import numpy as np
import cv2

#This example is for my personal pictures

# Fuse two color images.  Assume that zero indicates an unknown value.# At pixels where both values are known, the output is the average of the two.# At pixels where only one is known, the output uses that value.
def fuse_color_images(A, B):
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)
    C = np.zeros(A.shape, dtype=np.uint8)
    A_mask = np.sum(A, axis=2) > 0
    B_mask = np.sum(B, axis=2) > 0
    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask
    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]
    return C

#Mouse callback to detect points on the input images
#This was used to determine the points of the first image
def mouse_callback(event, x, y, flags, params):
    if event == 1:
        print(x,y)

#Initialize orb  and final render dimensions
orb = cv2.ORB_create()

render_height = 2500

render_width = 2500

#Initialize the list of input images
list_of_pics = [cv2.imread("pics/20201111_210559.jpg"), cv2.imread("pics/20201111_210604.jpg"), cv2.imread("pics/20201111_210611.jpg"), cv2.imread("pics/20201111_210617.jpg"), cv2.imread("pics/copy.jpg"),cv2.imread("pics/20201111_210630.jpg")]

image1 = cv2.imread("pics/20201111_210559.jpg")

#Points that I detected using mouse callback as well as ortho points I calculated by hand
#In the version where I used the museum photos the ortho values were calculated based on the pixel to cm given
pts1_ortho = np.array([[0,0], [500,0], [0, 500], [500,500]])

pts1 = np.array([[916,183], [2109,221], [878,1394], [2079,1372]])

#Compute the homagraphy matrix
H1, _ = cv2.findHomography(srcPoints=pts1, dstPoints=pts1_ortho)

H_Prev = H1

#apply perspective warp to new billboard
bgr_ortho = cv2.warpPerspective(image1, H1, (render_width, render_height))

#For each image and image ahead in the list. Use ORB to compute homagraphy. Then use this homagraphy and previous to get a final homagraphy used to stitch image to the ortho
for i in range(len(list_of_pics) - 1):
    #convert grayscale for ORB
    gray2 = cv2.cvtColor(list_of_pics[i], cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(list_of_pics[i+1], cv2.COLOR_BGR2GRAY)
    #Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #find points, match, and sort
    kpts1, descs1 = orb.detectAndCompute(gray1, None)
    kpts2, descs2 = orb.detectAndCompute(gray2, None)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    #compute homagraphy
    H,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #Compute final homagraphy for ortho
    H_Current = H_Prev @ H
    #Warp and stitch
    temp = cv2.warpPerspective(list_of_pics[i + 1], H_Current, (render_width, render_height))
    final = fuse_color_images(temp, bgr_ortho)
    bgr_ortho = final
    H_Prev = H_Current

#write file
cv2.imwrite("myoutput.jpg", bgr_ortho)
cv2.imshow('image', bgr_ortho)

# wait for a key to be pressed to exit
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()