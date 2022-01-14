import Palm_Tracking as pt
import cv2
import numpy as np
import os

headerList=[]

brushSize = 20
eraserSize = 75
# Setting up initial color
brushColor=(0,0,255)
# Creating static canvas
imgCanvas = np.zeros((480, 640, 3), np.uint8)

folder="Canvas"
list=os.listdir(folder)

xPrev, yPrev = 0, 0
# Fetching Images
for path in list:
    im=cv2.imread(f'{folder}/{path}')
    headerList.append(im)

header=headerList[3]
vCap=cv2.VideoCapture(0)
# Width
vCap.set(3,640)
# Height
vCap.set(4,480)

detector = pt.detectHand()

tipIds = [4, 8, 12, 16, 20]

while True:
    # Image capture
    success, img = vCap.read()
    # Flipping the image to reflect natural hand movements
    img=cv2.flip(img,1)

    # Fetching hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList)!=0:
        #print(lmList)
        # Co-ordinates of Index finger and Middle finger tips
        xI, yI = lmList[8][1],lmList[8][2]
        xM, yM = lmList[12][1],lmList[12][2]

        # Identifying which fingers are up/open
        fingers = []
        # Identifying if Thumb is open/up
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  # checking x position of 4 is in right to x position of 3
            fingers.append(1)
        else:
            fingers.append(0)
        # Identifying all the other fingers are open/up
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        #print(fingers)

        # If Index and Middle fingers are up then Selection Mode
        if fingers[1] and fingers[2] == True and fingers[3] == False:
            xPrev,yPrev=0,0
            #print("Selection Mode")
            # Identifying the selection (Red, Blue, Green or Eraser) based on the pixel values
            if yI < 70:
                if 145 < xI < 245:
                    brushColor = (0, 0, 255)
                    header = headerList[3]
                elif 270 < xI < 370:
                    brushColor = (255, 0, 0)
                    header = headerList[0]
                elif 400 < xI < 500:
                    brushColor = (0, 255, 0)
                    header = headerList[2]
                elif 520 < xI < 633:
                    brushColor = (0, 0, 0)
                    header = headerList[1]
            # If the fingers are below the header then display a rectangle stating the Selection mode
            cv2.rectangle(img, (xI, yI - 25), (xM, yM + 25), brushColor, cv2.FILLED)

        # If only the Index finger is up the Drawing Mode
        elif fingers[1] and fingers[2] == False :
            # Drawing mode is represented by circle
            cv2.circle(img, (xI, yI), 15, brushColor, cv2.FILLED)
            #print("Drawing Mode")
            # Initilizong the co-ordinates to draw from previous to current positions
            if xPrev == 0 and yPrev == 0:
                xPrev, yPrev = xI, yI

            # Identifying the Eraser based on the brush color selected. Erasing the content on video feed
            if brushColor == (0, 0, 0):
                # Drawing on both image captured from video and on a static image canvas
                cv2.line(img, (xPrev, yPrev), (xI, yI), brushColor, eraserSize)
                cv2.line(imgCanvas, (xPrev, yPrev), (xI, yI), brushColor, eraserSize)
            else:
                # If the eraser is not selected then draw using the brush color selected
                cv2.line(img, (xPrev, yPrev), (xI, yI), brushColor, brushSize)
                cv2.line(imgCanvas, (xPrev, yPrev), (xI, yI), brushColor, brushSize)

            xPrev,yPrev=xI,yI
        # Clear all when all the fingers are up
        elif all(x >= 1 for x in fingers):
            imgCanvas = np.zeros((480, 640, 3), np.uint8)

    # Merging the static canvas and the image captured from video feed to get the final result
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    # setting the header image
    img[0:70, 0:640] = header

    cv2.imshow("VS", img)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)