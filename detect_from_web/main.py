import cv2
import core.main as main

cap = cv2.VideoCapture("http://220.158.85.74:60001/cgi-bin/snapshot.cgi?chn=0&u=admin&p=&q=0&1630735710")
if cap.isOpened():
    ret,img = cap.read()
    cv2.imwrite("sample.jpg", img)
    p1 = main.Parking("demo1.xml", image="sample.jpg")
    p1.update_state_from_photo("sample.jpg")
    p1.draw_boxes()
    cv2.waitKey(0)