import math
import cv2

def draw_pet(frame, petX, petY):
    cv2.circle(frame, (petX, petY), 40, (255, 255, 255), -1)  #draws a circle where the pet is supposed to be

def is_near(x, y, x2, y2):
    return math.hypot(x2 - x, y2 - y) < 50 #checks if the given coordinate is close

