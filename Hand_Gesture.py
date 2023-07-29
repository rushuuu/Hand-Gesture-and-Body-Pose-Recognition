import cv2
import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handslms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handslms, mpHands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handslms.landmark:
                h, w, _ = img.shape
                lmx, lmy = int(lm.x * w), int(lm.y * h)
                landmarks.append((lmx, lmy))

    cv2.imshow("Hand Gesture Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
