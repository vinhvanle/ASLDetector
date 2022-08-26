import cv2
import time
import os
import uuid
import keyboard


hands = int(input("How many hands?"))
label = str(input("What is the label?"))
folder = os.path.join('Tensorflow','workspace','images')
if not os.path.exists(os.path.join(folder,label)):
    os.mkdir(os.path.join(folder,label))
image_path = os.path.join(folder,label)
cap = cv2.VideoCapture(0)
counter = 0

if hands == 1:
    while True:
        success, img = cap.read()
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{image_path}/{label}_{time.time()}.jpg', img)
            print(counter)
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
if hands == 2:
    number_img = int(input("How many pics?"))
    for imgnum in range(number_img):
        print('Collecting image {}'.format(imgnum))
        success, img = cap.read()
        cv2.imshow('Image', img)
        cv2.imwrite(f'{image_path}/{label}_{time.time()}.jpg',img)
        time.sleep(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()




