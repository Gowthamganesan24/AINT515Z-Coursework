import cv2
import numpy as np

cap = cv2.VideoCapture("coursework/v1.mp4")

# blob detector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.filterByConvexity = False
params.filterByInertia = False
params.filterByCircularity = False
params.thresholdStep = 10
params.minArea = 700
params.maxArea = 1400
count = 0
#To detect the blob
detector = cv2.SimpleBlobDetector_create(params)
while True:
    success, img = cap.read()
    if success != True:
        print('Successfully Executed')
        break
    cv2.imshow("Input", img)
    #Create Line For Circle Detection
    cv2.line(img, (280, 0), (280, 160), (255, 255, 255), 2)
    #converting the image from BGR to Gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Using the Gaussian bLur for better result
    gray_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #Coverting the BGR into HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert bgr to hsv image format
    lower_hsv = np.array([0, 20, 20])  # using the numpy to store the matrix hsv values of color
    upper_hsv = np.array([30, 255, 255])
    #masking the HSV to separate the needed blob and Wrap of the circle
    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    # Adaptive Threshold
    thresh = cv2.adaptiveThreshold(gray_blurred, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=5, C=20)
    #To detect the blob in the gray_blurred image
    blobs = detector.detect(gray_blurred)

    #To loop through all the blobs detected
    for i in blobs:
        cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), int(i.size/2), (255,0,100), -1)
        cv2.circle(img, (int(i.pt[0]), int(i.pt[1])), 3, (0, 255, 255), -1)
        # perform Hough transform to detect circles
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=60, param2=30, minRadius=15, maxRadius=30)
    # If the circles are detected
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # loop through list of circle to show all circles
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            x = center[0]

            # circle center
            cv2.circle(img, center, 1, (0, 0, 0), 3)
            # circle outline
            radius = circle[2]
            cv2.circle(img, center, radius, (0, 0, 255), 4)
            #To find the size of total blobs
            n = len(blobs)
            print(n)
            #creating the condition to count the blobs which crossed this specific point
            if 280 > x > 277:
                count = count + 1

    text = 'Number of detected :'
    g = str(count)
    o = text + g
    #To add the text in the final output video
    cv2.putText(img, str(o), (550, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    print(text)
    #cv2.imshow("res", gray_blurred)
    cv2.imshow("threshold", thresh)
    cv2.imshow("mask", mask)
    cv2.imshow("result", img)
    cv2.waitKey(10)
