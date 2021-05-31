import cv2
from telegramBot import send_detect_foto
import asyncio
import os

#загружаем каскады
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml')

#faces  = face_cascade.detectMultiScale(gray, 1.3, 5)

#определяем лицо и улыбку
async def detect_smile(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        #print(smiles)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
        if len(smiles) > 0:
            cv2.imwrite('graygirl.jpg', frame)
            if os.path.exists('graygirl.jpg'):
                #await asyncio.sleep(10)
                await send_detect_foto('graygirl.jpg')

    #возможно на будущее понадобиться работать с возвращенным кадром        
    return frame

