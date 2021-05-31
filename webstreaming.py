from motion_detection.single_motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Flask, Response, render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from smile_detect import detect_smile
import asyncio


outputFrame = None
lock = threading.Lock()

app = Flask(__name__)
#инициализация видеопотока 

#vs = VideoStream(usePiCamera=1).start()#если мы работаем с rasberryPu
vs = VideoStream(src=0).start()
#state = True
time.sleep(2.0)

@app.route("/")
def index():
    return render_template("index.html")

def detect_frame(frameCount):
    '''функция по работе с кадром из видеопотока'''
    # захват видеопотока, кадров
    global vs, outputFrame, lock
    #инициализируем наш класс по работе сдвижения и получения всех кадров
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    while True:
        #чтение последующих кадров входного потока, изменение размера и конвертирование в ч/б кадр для повышения производительности
        frame = vs.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)#фильтрация лишних контуров
        smile = asyncio.run(detect_smile(gray, frame))#вызов функции определения лица и улыбки - сохранение изображения с улыбкой, асинхронность для повышения скорости работы приложения
        # задаем время которое также будет отображаться
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime('%A %d %B %Y %I:%M:%S%p'), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        #проверка обнаружения движения
        if total > frameCount:
            #цикл задает частоту кадров, при которой отслеживается движение
            motion = md.detect(gray)
            #проверка обнаружения движения в кадре 
            if motion is not None:
                #распаковка кортежа и создание области вокруг зоны движения (рисуем прямоугольник)
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY), (0, 0, 255), 2)

        #обновляем фоновую модель и увеличиваем счетчик общего числа кадров
        md.update(gray)
        total += 1
		
        with lock:
            outputFrame = frame.copy()

def generate():
    '''функция отображения текущего кадра в окне браузера'''
    global outputFrame, lock, state

    while True:
        with lock:
			#проверка и подготовка кадра в нужном формате для отрисовки на странице браузера 
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
	#задание необходимых аргументов для работы функции детектирования в отдельном потоке
	ap = argparse.ArgumentParser()
	#ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
	#ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32, help="# of frames used to construct the background model")
	args = vars(ap.parse_args())
	#старт второго потока, котрый отслеживает движение в кадре
	t = threading.Thread(target=detect_motion, args=(args["frame_count"],))
	t.daemon = True
	t.start()
	#запуск приложения
	app.run( debug=True, threaded=True, use_reloader=False)
#закрытие видеопотока
vs.stop()
