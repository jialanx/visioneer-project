import cv2
import mediapipe as mp

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

    cv2.imshow("Visioneer", frame) #Shows frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()