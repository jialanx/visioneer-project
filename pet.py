import math
import cv2

def draw_pet(frame, posX, posY):
    cv2.circle(frame, (posX, posY), 40, (255, 255, 255), -1)  #draws a circle where the pet is supposed to be

def is_near(x, y, x2, y2, dist):
    return math.hypot(x2 - x, y2 - y) < dist #checks if the given coordinate is within distance

