import cv2 as cv
import numpy as np


PIXEL_TO_MM = 0.5

def ConvertMM(contour, pixel_to_mm):
    if contour is None or len(contour) == 0:
        return 0.0
    perimeter_px = cv.arcLength(contour, True)
    perimeter_mm = perimeter_px * pixel_to_mm
    return perimeter_mm

def DetectColor(frame, Lower, Upper, COLORNAME='', pixel_to_mm=PIXEL_TO_MM):
    ColorCounter = 0
    FrameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    blur = cv.GaussianBlur(FrameHSV, (3, 3), 0)
    mask = cv.inRange(blur, Lower, Upper)
    kernel = np.ones((5, 5), np.uint8)
    Resmask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    contours, _ = cv.findContours(Resmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    BGR = (255, 100, 10)
    for cnt in contours:
        if cv.contourArea(cnt) > 600:
            perimeter_mm = int(round(ConvertMM(cnt, pixel_to_mm)))
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame, [box], 0, BGR, 2)
            ColorCounter += 1
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(frame, (cx, cy), 5, BGR, -1)
                cv.putText(frame,f"({cx},{cy})",(cx,cy),cv.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),2)

                box_sorted = box[np.argsort(box[:, 1])]
                bottom_points = box_sorted[-2:]
                bottom_center_x = int((bottom_points[0][0] + bottom_points[1][0]) / 2)
                bottom_center_y = int((bottom_points[0][1] + bottom_points[1][1]) / 2)
                cv.putText(frame, f"{perimeter_mm} mm", (bottom_center_x - 30, bottom_center_y + 20),cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 2)

                x, y, w, h = cv.boundingRect(cnt)
                cv.putText(frame, COLORNAME, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.4, BGR, 2)
    return ColorCounter


cap = cv.VideoCapture(0)
while True:
    ret, cam = cap.read()
    BLUE = DetectColor(cam, np.array([100, 150, 50]), np.array([135, 255, 255]), "BLUE")
    GREEN = DetectColor(cam, np.array([40, 50, 50]), np.array([90, 255, 255]), "GREEN")
    if BLUE or GREEN:
        print(f"BLUE : {BLUE} , GREEN : {GREEN}")
    cv.imshow('WEBCAM', cam)
    if cv.waitKey(1) == 27:  
        break
cap.release()
cv.destroyAllWindows()
