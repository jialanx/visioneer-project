import cv2
import mediapipe as mp
import random
import time
from collections import deque
import math 
import simpleaudio as sa

# loads assets
bird_sit = cv2.imread("sprites/birdsit.png", cv2.IMREAD_UNCHANGED)
bird_one = cv2.imread("sprites/birdflyone.png", cv2.IMREAD_UNCHANGED)
bird_two = cv2.imread("sprites/birdflytwo.png", cv2.IMREAD_UNCHANGED)
bird_peek_one = cv2.imread("sprites/birdlookone.png", cv2.IMREAD_UNCHANGED)
bird_peek_two = cv2.imread("sprites/birdlooktwo.png", cv2.IMREAD_UNCHANGED)
branch = cv2.imread("sprites/branch.png", cv2.IMREAD_UNCHANGED)
bird_chirp = cv2.imread("sprites/bird_chirp_audio.png", cv2.IMREAD_UNCHANGED)
bird_chirp_audio= sa.WaveObject.from_wave_file("audio/bird_chirp_audios.wav")

# resize is needed
bird_sit = cv2.resize(bird_sit, (200, 200), interpolation = cv2.INTER_AREA)
bird_chirp = cv2.resize(bird_chirp, (200, 200), interpolation = cv2.INTER_AREA)
bird_one = cv2.resize(bird_one, (200, 200), interpolation = cv2.INTER_AREA)
bird_two = cv2.resize(bird_two, (200, 200), interpolation = cv2.INTER_AREA)
branch = cv2.resize(branch, (300, 300), interpolation = cv2.INTER_AREA)

def is_near(x, y, x2, y2, dist):
    return math.hypot(x2 - x, y2 - y) < dist # checks if the given coordinate is within distance

def overlay_image_alpha(original, overlay, x, y):
    h, w = overlay.shape[:2] # height and width of overlay (:2 is first 2 elements)
    og_h, og_w = original.shape[:2]

    # bounds for overlay and original shapes
    # this is needed for alpha blending below and making sure
    # the image displayed is within the screen
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + w, og_w)
    y2 = min(y + h, og_h)

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    # alpha blending (this allowed transparent pixels to show!)
    alpha = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0 # grabs alpha channel, makes it a value between 0-1
    for c in range(3):  # does alpha blending for each pixel
        original[y1:y2, x1:x2, c] = (
            alpha * overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, c] +
            (1 - alpha) * original[y1:y2, x1:x2, c]
        )

# sets up variables
swipe_coords = deque(maxlen=8) # stores last 5 posX (for swiping)
swipe_cooldown = 0
last_seen_face= 0
last_close_up = 0
animation_delay = 0
sprite = bird_one
current_sprite = "one"
audio_cooldown = 0
mode = "fly"

cap = cv2.VideoCapture(0) # opens webcam

mp_face = mp.solutions.face_detection # sets up face detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) # will detect if it is > 50% sure it is a face

mp_hands = mp.solutions.hands # sets up hand detection
hands = mp_hands.Hands() # tracks hands
mp_draw = mp.solutions.drawing_utils # draws tracking lines
bird_posX, bird_posY = 300, 300 # initial pet position

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

    if mode == "fly":
        if time.time() - animation_delay > 0.2: # alternates sprites for flying animation
            if current_sprite == "one":
                sprite = bird_two
                current_sprite = "two"
            else:
                sprite = bird_one
                current_sprite = "one"
            animation_delay = time.time()
            if vx > 0: # if the velocity is negative, flip the sprite
                sprite = cv2.flip(sprite, 1)

    if face_results.detections: 
        last_seen_face = time.time() # updates the latest face seen time
        face_visible = True 
        for detection in face_results.detections:
            box = detection.location_data.relative_bounding_box # bounding box
            face_area = box.width * box.height 

            if face_area > 0.7: # if the area of the face is above 70% of the screen
                face_close = True
                last_close_up = time.time()
    
    if face_close and time.time() - last_close_up < 8: # if the face has been close up for over 5s
        mode = "peek"
        if time.time() - animation_delay > 0.2: # play the animation for a close up
            vx = 0
            vy = 0
            bird_posX = 320
            bird_posY = 240
            if current_sprite == "one":
                sprite = bird_peek_two
                current_sprite = "two"
            else:
                sprite = bird_peek_one
                current_sprite = "one"
            animation_delay = time.time()

    elif mode == "peek" and not face_close:
        mode = "fly"

    if not face_visible and time.time() - last_seen_face > 7: # if the face has been gone for over 7s
        if bird_posX < 550: # if not already, the bird will fly towards top right branch
                vx = 5
        if bird_posY > 60:
                vy = -3
        if is_near(bird_posX, bird_posY, 550, 60, 40): # once the bird is close, it will sit on the branch and chirp
            bird_posX = 550
            bird_posY = 60
            mode = "chirp"
            sprite = bird_chirp
            if time.time() - audio_cooldown > 2:
                bird_chirp_audio.play()
                audio_cooldown = time.time()
    elif mode == "chirp":
        mode = "fly"


    if result.multi_hand_landmarks: # if there are hand landmarks
        for hand_landmarks in result.multi_hand_landmarks: # go through each of them
            #mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) # draw it
            for id, lm in enumerate(hand_landmarks.landmark): # for each id (dot) give their (lm = landmark / coordinates)
                h, w, _ = frame.shape 
                cx, cy = int(lm.x * w), int(lm.y * h) # pixel positions on screen

                if id == 8: # index fingertip
                    #cv2.circle(frame, (cx, cy), 10, (255,0,0),cv2.FILLED) # draw blue dot on index finger
                    swipe_coords.append(cx) # add the most recent X to swipe coords
                    if is_near(cx,cy,bird_posX,bird_posY,15):
                        if time.time() - audio_cooldown > 4:
                            bird_chirp_audio.play()
                            audio_cooldown = time.time()

                # checks if fingers are extended
                extended_finger_id = [(8, 6), (12, 10), (16,14), (20, 18)] # fingers and the joint that is directly below them
                extended_fingers = 0
                for tip, middle in extended_finger_id: 
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[middle].y: # checks if the tip of the finger is above the joint
                        extended_fingers += 1
                
                if extended_fingers == 4: # if the 4 fingers are up, the palm is out
                    
                    # shorten for readability
                    lm6 = hand_landmarks.landmark[6]
                    lm10 = hand_landmarks.landmark[10]
                    lm14 = hand_landmarks.landmark[14]

                    if abs(lm6.x - lm10.x) < 0.04 and abs(lm10.x - lm14.x) < 0.04: # check if these joints are overlapping on screen (this means the palm is facing sideways)
                        #cv2.circle(frame, (int(lm6.x * w), int(lm6.y * h)), 10, (0,0,255),cv2.FILLED) # draws red circle where joints overlap

                        if len(swipe_coords) == 8:
                            if all(swipe_coords[i] < swipe_coords[i+1] for i in range(len(swipe_coords)-1)): # makes sure they are all going in the same direction
                                if time.time() - swipe_cooldown > 1: # its been at >1s since last swipe
                                    swipe_coords.clear()
                                    swipe_cooldown = time.time()
                                    if (bird_posX > cx) and (is_near(bird_posX, bird_posY, cx, cy, 200)): # if the pet is near, change the velocity "blowing it away"
                                        vx = 7

                            elif all(swipe_coords[i] > swipe_coords[i+1] for i in range(len(swipe_coords)-1)):
                                if time.time() - swipe_cooldown > 1:
                                    swipe_coords.clear()
                                    swipe_cooldown= time.time()
                                    if (bird_posX < cx) and (is_near(bird_posX, bird_posY, cx, cy, 200)):
                                            vx = -7

                elif id == 6: # middle of index finger
                    #cv2.circle(frame, (cx, cy), 10, (0,255,0),cv2.FILLED) # draw red dot on index far
                    if is_near(cx, cy, bird_posX, bird_posY, 50):
                        bird_posX, bird_posY = cx, cy # it touching, bird will 'perch'
                        sprite = bird_sit

    overlay_image_alpha(frame, branch, 350, -30) # display branch
    overlay_image_alpha(frame, sprite, bird_posX - sprite.shape[1] // 2, bird_posY - sprite.shape[0] // 2) # display the bird (make sure image is centered)

    cv2.imshow("Visioneer", frame) # shows frame
    if cv2.waitKey(1) & 0xFF == ord('q'): # closes loop when q is pressed
        break

    # changes velocity at a random point ~60s
    if random.randint(0,60) == 0:
        vx = random.choice([-2, -1, 1, 2])
        vy = random.choice([-2, -1, 1, 2])
    bird_posX += vx
    bird_posY += vy

    # if on the frame, bounce
    if bird_posX < 40 or bird_posX > frame_width - 40:
        vx *= -1
    if bird_posY < 40 or bird_posY > frame_height - 40:
        vy *= -1

cap.release() # close lens
cv2.destroyAllWindows() # terminate program