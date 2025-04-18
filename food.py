import cv2

# draws food
def draw_food(frame, posX, posY):
    cv2.circle(frame, (posX, posY), 40, (255, 0, 255), -1)  



