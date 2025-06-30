import cv2 as cv 
import numpy as np 

def draw_Contours(frame,mask,BGR,name,count):
    contours,_ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt)>600:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame,[box],0,BGR,2)
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv.circle(frame,(cx,cy),5,BGR,-1)
            x,y,w,h = cv.boundingRect(cnt)
            cv.putText(frame,name,(x,y-10),cv.FONT_HERSHEY_COMPLEX,0.5,BGR,2)
            count+=1
    return count

def gaussian_blur(frame,kernel_size=(3,3),sigma=0):
    blur= cv.GaussianBlur(frame,ksize=kernel_size,sigmaX=sigma)
    return blur

def detect_color(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    blur = gaussian_blur(hsv)

    lowR1 = np.array([0,120,70])
    upR1 = np.array([10,255,255])
    
    lowR2 = np.array([170,120,70])
    upR2 = np.array([180,255,255])

    lowG=np.array([40,50,50])
    upG = np.array([90,255,255])

    lowB = np.array([100,150,50])
    upB = np.array([135,255,255])


    maskR1 = cv.inRange(blur,lowR1,upR1)
    maskR2 = cv.inRange(blur,lowR2,upR2)
    maskREDres = cv.add(maskR1,maskR2)


    maskG = cv.inRange(blur,lowG,upG)


    mask_B = cv.inRange(blur,lowB,upB)

    kernel = np.ones((5,5),np.uint8)
    mask_R = cv.morphologyEx(maskREDres,cv.MORPH_OPEN,kernel)
    mask_G = cv.morphologyEx(maskG,cv.MORPH_OPEN,kernel)

    red_count = 0
    green_count = 0
    blue_count = 0
    green_count = draw_Contours(frame,mask_G,(0,255,0),"GREEN",green_count)
    red_count = draw_Contours(frame,mask_R,(0,0,255),"RED",red_count)
    blue_count = draw_Contours(frame,mask_B,(255,0,0),"BLUE",blue_count)
    return frame,red_count,green_count,blue_count

def get_center(contour):
    x,y,w,h = cv.boundingRect(contour)
    cx = x+w //2
    cy = y+h //2

    return cx,cy 
 

cap = cv.VideoCapture(0)
while(True):
    ret,cam=cap.read()
    res,red,green,blue = detect_color(cam)
    print(f"red : {red} , green : {green}, blue : {blue}")
    cv.imshow('WEBCAM',cam)
    if cv.waitKey(1)== 27 :
        break
cv.destroyAllWindows()