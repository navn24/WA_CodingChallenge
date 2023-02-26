#Author: Naveen Balu

'''
General Approach: Create Mask to Detect Red Objects (Cones) on the left and right sides of the image
then get their coordinates, fit a line on those coordinates, and plot those lines
NOTE: Lots of code is commented out because it highlights the different methods I had tried prior
'''
 
import cv2
import numpy as np
import imutils

# Reading the image
img = cv2.imread('red.png')


# lower bound and upper bound for Red color

lower_bound = np.array([10, 10, 175])
upper_bound = np.array([36, 56, 216])

# find the colors within the boundaries
mask = cv2.inRange(img, lower_bound, upper_bound)

#define kernel size  
kernel = np.ones((7,7),np.uint8)

# Remove unnecessary noise from mask
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# output = cv2.bitwise_and(img, img, mask = mask)

# Find contours from the mask

contours,hier = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# contours = imutils.grab_contours(contours)

rows,cols = img.shape[:2]

left = []
right = []
#Get the points of each of the cones and separate them based on their left or right positions
for c in contours:
    # compute the center of the contour
    if cv2.contourArea(c) == 0:
        continue
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # points.append([cX,cY])
    # points = np.array(points)
    if cX<=cols/2:
        left.append([cX,cY])
        
        # point = np.array([cX,cY])
        # print((cX,cY))
        # draw the contour and center of the shape on the image
        # i = cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        # img = cv2.circle(img, (cX,cY), radius=2, color=(0, 255, 0), thickness=-1)
    else:
        right.append([cX,cY])
left = np.array(left)
right = np.array(right)

m=100000

#Draw the left line
vx,vy,x0,y0 = cv2.fitLine(left, cv2.DIST_L2,0,0.01,0.01)
vx,vy,x0,y0=vx[0],vy[0],x0[0],y0[0]

start = (int(x0-m*vx), int(y0-m*vy))
end = (int(x0+m*vx), int(y0+m*vy))
cv2.line(img, start, end, (0,0,255),2)

#Draw the right line
vx,vy,x0,y0 = cv2.fitLine(right, cv2.DIST_L2,0,0.01,0.01)
vx,vy,x0,y0=vx[0],vy[0],x0[0],y0[0]

start = (int(x0-m*vx), int(y0-m*vy))
end = (int(x0+m*vx), int(y0+m*vy))
cv2.line(img, start, end, (0,0,255),2)

# print(type(left),type(left[0]))
# p1 = left[0]
# p2 = left[-1]

# # theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
# # endpt_x = int(p1[0] - 1000*np.cos(theta))
# # endpt_y = int(p1[1] - 1000*np.sin(theta))
# vx,vy = p2[0]-p1[0],p2[1]-p1[1]
# # p2[0] = 1000*(p2[0]-p1[0])
# # p2[1] = 1000*(p2[1]-p1[1])
# m=100
# img = cv2.line(img,(p1[0],p1[1]),(p1[0]+m*vx,p1[1]+m*vy),(0,0,255))

# img = cv2.circle(img, (p1[0],p1[1]), radius=2, color=(0, 255, 0), thickness=-1)
# img = cv2.circle(img, (p2[0],p2[1]), radius=2, color=(0, 255, 0), thickness=-1)

# p1 = right[0]
# p2 = right[-1]
# vx,vy = p2[0]-p1[0],p2[1]-p1[1]

# img = cv2.line(img,(p1[0],p1[1]),(p1[0]+m*vx,p1[1]+m*vy),(0,0,255))
# img = cv2.circle(img, (p1[0],p1[1]), radius=2, color=(0, 255, 0), thickness=-1)
# img = cv2.circle(img, (p2[0],p2[1]), radius=2, color=(0, 255, 0), thickness=-1)

# print(left)

# vx,vy,x0,y0 = cv2.fitLine(left, cv2.DIST_L2,0,0.01,0.01)
# # print(vx,vy,x0,y0)
# vx,vy,x0,y0=vx[0],vy[0],x0[0],y0[0]

# # print(vx,vy,x0,y0)
# m=100000
# start = (int(x0-m*vx), int(y0-m*vy))
# end = (int(x0+m*vx), int(y0+m*vy))
# cv2.line(img, start, end, (0,0,255),2)

# vx,vy,x0,y0 = cv2.fitLine(right, cv2.DIST_L2,0,0.01,0.01)
# # print(vx,vy,x0,y0)
# vx,vy,x0,y0=vx[0],vy[0],x0[0],y0[0]

# # print(vx,vy,x0,y0)
# m=100000
# start = (int(x0-m*vx), int(y0-m*vy))
# end = (int(x0+m*vx), int(y0+m*vy))
# cv2.line(img, start, end, (0,0,255),2)

#     lefty = int((-x*vy/vx) + y)
#     righty = int(((cols-x)*vy/vx)+y)
#     print(vx,vy,x,y,lefty,righty)

#     img = cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

# print(points)


cv2.imshow('image',img)
cv2.imwrite('output.png',img)
# output = cv2.drawContours(output, contours, -1, (0, 0, 255), 3)

# cv2.imshow("images", np.hstack([img, output]))
# cv2.imshow("images", output)


cv2.waitKey(0)
cv2.destroyAllWindows()
