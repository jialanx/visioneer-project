import cv2

# draws pet
def draw_pet(frame, posX, posY):
    cv2.circle(frame, (posX, posY), 40, (255, 255, 255), -1)  #draws a circle where the pet is supposed to be

