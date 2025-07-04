import cv2 as cv 
import numpy as np

def DetectColor(frame,Lower,Upper,COLORNAME=''):
    ColorCounter=0
    FrameHSV = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    blur= cv.GaussianBlur(FrameHSV,ksize=(3,3),sigmaX=0)
    mask = cv.inRange(blur,Lower,Upper)
    kernel = np.ones((5,5),np.uint8)
    Resmask = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel)
    contours,_ = cv.findContours(Resmask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    BGR = (50,255,0)
    for cnt in contours:
        if cv.contourArea(cnt)>600:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame,[box],0,BGR,2)
            ColorCounter+=1
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv.circle(frame,(cx,cy),5,BGR,-1)
                cv.putText(frame,f"x:{cx},y:{cy}",(cx+10,cy-10),cv.FONT_HERSHEY_COMPLEX,0.5,(255,255,0),2)
                x,y,w,h = cv.boundingRect(cnt)
                cv.putText(frame,COLORNAME,(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.4,BGR,2)
    return ColorCounter


cap = cv.VideoCapture(0)
while(True):
    ret,cam=cap.read()
    BLUE = DetectColor(cam,np.array([100,150,50]),Upper=np.array([135,255,255]),COLORNAME="BLUE")
    GREEN = DetectColor(cam,np.array([40,50,50]),Upper=np.array([90,255,255]),COLORNAME="GREEN")
    if BLUE or GREEN:
        print(f"BLUE : {BLUE} ,GREEN : {GREEN}")
    cv.imshow('WEBCAM',cam)
    if cv.waitKey(1)== 27 :
        break
cv.destroyAllWindows()
