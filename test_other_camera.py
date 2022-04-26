import cv2

cap = cv2.VideoCapture("/dev/video0")

print(cap.isOpened())