import cv2
import numpy as np
import time
import math
import mediapipe as mp

class detectHand():
    def __init__(self,mode=False,maxHands=1,modelComplexity=1,detectionCon=0.5,trackCon=0.5):
        # Initial variables setup
        self.mode=mode 
        self.maxHands=maxHands
        self.modelComplex = modelComplexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        # initializing instance of Media pipe hands module
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detectionCon,self.trackCon) #object for Hands for a particular instance
        # Creating istance of Drawing object
        self.mpDraw=mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        # converting from Blue, Green, Red to Red, Green, Blue as all the methods wok with RGB
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #processing the RGB image
        self.results=self.hands.process(imgRGB)
        # retrives x,y,z values of every landmark
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:#each hand landmarks in results
                if draw:
                    # Drawing handlandmarks along with the connections on the detected hand
                     self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self,img,handNo=0):
        self.lmlist=[]
        if self.results.multi_hand_landmarks:
            # Fetch results of perticular hand
            myHand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                # Retriving height ,width to convert landmarks results x,y into pixel values
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id, cx, cy)
                self.lmlist.append([id,cx,cy])

        return self.lmlist