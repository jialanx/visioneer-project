import cv2
import mediapipe as mp
import random
import time
from collections import deque
from pet import draw_pet
from food import draw_food
import math 

def is_near(x, y, x2, y2, dist):
    return math.hypot(x2 - x, y2 - y) < dist # checks if the given coordinate is within distance

swipe_coords = deque(maxlen=8) # stores last 5 posX (for swiping)
swipe_cooldown = 0
last_seen_face= 0
last_close_up = 0

cap = cv2.VideoCapture(0) # opens webcam

mp_face = mp.solutions.face_detection # sets up face detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) # will detect if it is > 50% sure it is a face

mp_hands = mp.solutions.hands # sets up hand detection
hands = mp_hands.Hands() # tracks hands
mp_draw = mp.solutions.drawing_utils # draws tracking lines
petX, petY = 300, 300 # initial pet position
foodX, foodY = 50, 50 # initial food position

vx, vy = random.choice([-2, 2]), random.choice([-2, 2]) # set initial pet velocity

# frame
frame_width = 640
frame_height = 480

# when the program is running
while True: 
    success, frame = cap.read() # grabs a frame, checks if it works - if not, shut down program
    if not success: 
        break

    frame = cv2.flip(frame, 1) # flips camera horizontally so it looks right
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # converts BGR to RGB (hand coloring)
    result = hands.process(rgb) 
    face_results = face_detection.process(rgb) # detects face
    face_visible = False # is the face in vision?
    face_close = False # is the face close up?

    if face_results.detections: 
        last_seen_face = time.time() # updates the latest face seen time
        face_visible = True 
        for detection in face_results.detections:
            box = detection.location_data.relative_bounding_box # bounding box
            face_area = box.width * box.height 

            if face_area > 0.8: # if the area of the face is above 80% of the screen
                face_close = True
                last_close_up = time.time()
    
    if face_close and time.time() - last_close_up < 5: # if the face has been close up for over 5s
        print("face is up close")
    if not face_visible and time.time() - last_seen_face > 7: # if the face has been gone for over 7s
        print("face is gone")

    if result.multi_hand_landmarks: # if there are hand landmarks
        for hand_landmarks in result.multi_hand_landmarks: # go through each of them
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # draw it
            for id, lm in enumerate(hand_landmarks.landmark): # for each id (dot) give their (lm = landmark / coordinates)
                h, w, _ = frame.shape 
                cx, cy = int(lm.x * w), int(lm.y * h) # pixel positions on screen

                if id == 8: # index fingertip
                    cv2.circle(frame, (cx, cy), 10, (255,0,0),cv2.FILLED) # draw blue dot on index finger
                    if is_near(cx,cy,petX,petY,50):
                        cv2.putText(frame, "close", (petX-70, petY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # says close if finger is close
                    
                # checks if fingers are extended
                extended_finger_id = [(8, 6), (12, 10), (16,14), (20, 18)] # fingers and the joint that is directly below them
                extended_fingers = 0
                for tip, middle in extended_finger_id: 
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[middle].y: # checks if the tip of the finger is above the joint
                        extended_fingers += 1
                
                if extended_fingers == 4: # if the 4 fingers are up, the palm is out
                    cv2.putText(frame, "palm out", (petX-70, petY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) 
                    
                    # shorten for readability
                    lm6 = hand_landmarks.landmark[6]
                    lm10 = hand_landmarks.landmark[10]
                    lm14 = hand_landmarks.landmark[14]

                    if abs(lm6.x - lm10.x) < 0.04 and abs(lm10.x - lm14.x) < 0.04: # check if these joints are overlapping on screen (this means the palm is facing sideways)
                        cv2.circle(frame, (int(lm6.x * w), int(lm6.y * h)), 10, (0,0,255),cv2.FILLED)
                    
                        swipe_coords.append(cx) # add the most recent X to swipe coords

                        if len(swipe_coords) == 8:
                            if all(swipe_coords[i] < swipe_coords[i+1] for i in range(len(swipe_coords)-1)): # makes sure they are all going in the same direction
                                if time.time() - swipe_cooldown > 1: # its been at >1s since last swipe
                                    print("swipe right")
                                    swipe_coords.clear()
                                    swipe_cooldown = time.time()
                                    if (petX > cx) and (is_near(petX, petY, cx, cy, 200)): # if the pet is near, change the velocity "blowing it away"
                                        print("blown away")
                                        vx = 6

                            elif all(swipe_coords[i] > swipe_coords[i+1] for i in range(len(swipe_coords)-1)):
                                if time.time() - swipe_cooldown > 1:
                                    print("swipe left")
                                    swipe_coords.clear()
                                    swipe_cooldown= time.time()
                                    if (petX < cx) and (is_near(petX, petY, cx, cy, 200)):
                                            print("blown away")
                                            vx = -6

                elif id == 6: # middle of index finger
                    cv2.circle(frame, (cx, cy), 10, (0,255,0),cv2.FILLED) # draw red dot on index far
                    if is_near(cx, cy, petX, petY, 50):
                        petX, petY = cx, cy # it touching, bird will 'perch'

    draw_pet(frame, petX, petY) # draw the pet
    draw_food(frame, foodX, foodY) # draw the food

    if is_near(petX, petY, foodX, foodY, 80):
        cv2.putText(frame, "eating", (petX-70, petY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # bird is eating from feeder

    cv2.imshow("Visioneer", frame) # shows frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # closes loop when q is pressed
        break

    # changes velocity at a random point ~60s
    if random.randint(0,60) == 0:
        vx = random.choice([-2, -1, 1, 2])
        vy = random.choice([-2, -1, 1, 2])
    petX += vx
    petY += vy

    # if on the frame, bounce
    if petX < 40 or petX > frame_width - 40:
        vx *= -1
    if petY < 40 or petY > frame_height - 40:
        vy *= -1

cap.release() # close lens
cv2.destroyAllWindows() # terminate program