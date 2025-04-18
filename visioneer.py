import cv2
import mediapipe as mp
import random
import time
from collections import deque
from pet import draw_pet, is_near
from food import draw_food

swipe_buffer = deque(maxlen=8) #stores last 5 posX (for swiping)
swipe_cooldown = 0

cap = cv2.VideoCapture(0) #opens webcam

mp_hands = mp.solutions.hands
hands = mp_hands.Hands() #loads hand tracking module
mp_draw = mp.solutions.drawing_utils #draws tracking lines
petX, petY = 300, 300
foodX, foodY = 50, 50

vx, vy = random.choice([-2, 2]), random.choice([-2, 2])

frame_width = 640
frame_height = 480

while True:
    success, frame = cap.read() #grabs a frame, checks if it works
    if not success: #if it doesnt work stop the code
        break

    frame = cv2.flip(frame, 1) #flips camera horizontally so it looks right
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converts BGR to RGB (coloring)

    result = hands.process(rgb) 
    if result.multi_hand_landmarks: #Checks if it found hands
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #Draws the hand
            for id, lm in enumerate(hand_landmarks.landmark): #for each id (dot) give their (lm = landmark / coordinates)
                h, w, _ = frame.shape #pixel positions
                cx, cy = int(lm.x * w), int(lm.y * h) #multiples the coordinate to the pixel position

                if id == 8: #fingertip
                    cv2.circle(frame, (cx, cy), 10, (255,0,0),cv2.FILLED) #draw blue dot on index finger
                    swipe_buffer.append(cx)

                    if len(swipe_buffer) == 8:
                        if all(swipe_buffer[i] < swipe_buffer[i+1] for i in range(len(swipe_buffer)-1)):
                                if time.time() - swipe_cooldown > 1:
                                    print("swipe right")
                                    swipe_cooldown = time.time()

                        elif all(swipe_buffer[i] > swipe_buffer[i+1] for i in range(len(swipe_buffer)-1)):
                            if time.time() - swipe_cooldown > 1:
                                print("swipe left")
                                swipe_cooldown= time.time()


                    if is_near(cx,cy,petX,petY,50):
                        cv2.putText(frame, "close", (petX-70, petY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) #says close if finger is close
                    
                if id == 6:
                    cv2.circle(frame, (cx, cy), 10, (0,255,0),cv2.FILLED) #draw red dot on index far
                    if is_near(cx,cy,petX,petY,50):
                        petX, petY = cx, cy #if on the long part of the index finger, it will follow
                    
                #checks if the palm is out
                extended_finger_id = [(8, 6), (12, 10), (16,14), (20, 18)] #fingers and the joint that is directly below them
                extended_fingers = 0
                for tip, middle in extended_finger_id: #checks if the tip of the finger is above the joint
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[middle].y:
                        extended_fingers += 1
                
                if extended_fingers == 4: #if the 4 fingers (nothumb) are up, the palm is out
                    cv2.putText(frame, "palm out", (petX-70, petY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) #says close if finger is close
                    if (id == 6):
                        lm6 = hand_landmarks.landmark[6]
                        lm10 = hand_landmarks.landmark[10]
                        lm14 = hand_landmarks.landmark[14]

                        if abs(lm6.x - lm10.x) < 0.04 and abs(lm10.x - lm14.x) < 0.04:
                            cv2.circle(frame, (int(lm6.x * w), int(lm6.y * h)), 10, (0,0,255),cv2.FILLED)

    draw_pet(frame, petX, petY)
    draw_food(frame, foodX, foodY)

    if is_near(petX, petY, foodX, foodY, 80):
        cv2.putText(frame, "eating", (petX-70, petY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) #says close if finger is close
    if is_near(petX, petY, foodX, foodY, 30):
        cv2.putText(frame, "suffocating", (petX-70, petY-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) #says close if finger is close

    cv2.imshow("Visioneer", frame) #Shows frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #closes app when q is pressed
        break

    if random.randint(0,60) == 0:
        vx = random.choice([-2, -1, 1, 2])
        vy = random.choice([-2, -1, 1, 2])

    petX += vx
    petY += vy

    if petX < 40 or petX > frame_width - 40:
        vx *= -1
    if petY < 40 or petY > frame_height - 40:
        vy *= -1

cap.release()
cv2.destroyAllWindows()