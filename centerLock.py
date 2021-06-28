import cv2
import numpy as np
import imutils



frame = cv2.imread('red.png')
    # It converts the BGR color space of image to HSV color space
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold of blue in HSV space
lower_blue = np.array([0, 0, 255]) #175
upper_blue = np.array([179, 255, 255])

# preparing the mask to overlay
mask = cv2.inRange(hsv, lower_blue, upper_blue)
##################################################################
#DILATION

kernel = np.ones((5, 5), np.uint8)
dilated_img = cv2.dilate(mask, kernel, iterations=2)
cv2.imshow("filled gaps for contour detection", dilated_img)
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv2.bitwise_and(frame, frame, mask = mask)

# find contours in the thresholded image
###################################################################
cnts = cv2.findContours(dilated_img.copy(), cv2.RETR_EXTERNAL,
                         cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
# compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # draw the contour and center of the shape on the image
    cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
    cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(frame, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # show the image
    cv2.imshow("Image", frame)
    cv2.waitKey(0)
cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow('result', result)

cv2.waitKey(0)

cv2.destroyAllWindows()
