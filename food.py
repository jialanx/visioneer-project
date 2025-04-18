import cv2

def draw_food(frame, posX, posY):
    cv2.circle(frame, (posX, posY), 40, (255, 0, 255), -1)  #draws a circle where the food is supposed to be



