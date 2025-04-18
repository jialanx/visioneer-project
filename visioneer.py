import cv2
import mediapipe as mp
from pet import draw_pet, is_near

cap = cv2.VideoCapture(0) #opens webcam

mp_hands = mp.solutions.hands
hands = mp_hands.Hands() #loads hand tracking module
mp_draw = mp.solutions.drawing_utils #draws tracking lines

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
                print(f"Index coord: x={cx}, y={cy}")

                if is_near(cx,cy,petX,petY):
                    cv2.putText(frame, "close", (petX-70, petY-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) #says close if finger is close

    petX, petY = 300, 300
    draw_pet(frame, petX, petY)


    cv2.imshow("Visioneer", frame) #Shows frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #closes app when q is pressed
        break

cap.release()
cv2.destroyAllWindows()