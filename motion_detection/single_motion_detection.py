import numpy as np
import imutils
import cv2

#класс по работе с обработкой движения в кадре (вычисление разницы между предыдущим и текущим кадрами)
class SingleMotionDetector:
    def __init__(self, accumWeight=0.5):
        #хранение накопленного весового коэффицента
        self.accumWeight = accumWeight
		#инициализация модели бэкграунда
        self.bg = None

    def update(self, image):
        '''метод cv2.accumulateWeighted вычисляет взвешенную сумму входного изображения img и накопителя self.bg таким образом,
        что self.bg становится средним значением последовательности всех полученных кадров, чтобы при следующем кадре можно было
        фиксировать изменение пикселей, а следовательно фиксировать движение. Использует метод скользящего среднего'''
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        #обновление модели кадра
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def detect(self, image, tVal=25):
        '''вычисление абсолютной разницы между фоновой моделью
         и переданным изображением, а затем предел значения дельта-изображения'''
        delta = cv2.absdiff(self.bg.astype("uint8"), image)#нахождение разницы двух кадров, которая проявляется лишь при изменении одного из них
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]#метод для выделения кромки объекта белым цветом, чтобы потом найти контуры
        #эрозия и делотация
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)#расширяет выделенную на предыдущем этапе область

        #найдите контуры в пороговом изображении и инициализируйте область в которой будет происходить движение
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#нахождение массива контурных точек
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)#формируем область в которой происходило движение
        (maxX, maxY) = (-np.inf, -np.inf)

        #если контуры не найдены, возвращаем None
        if len(cnts) == 0:
            return None
        #в противном случае проходим цикл на контурах
        for c in cnts:
            #находим ограничительные рамки каждого контура:
            #начальные x и y координаты, за которыми следуют ширина и высота рамки
            #обновление координат ограничивающей движение области
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x), min(minY, y))#обновление области, в которой происходило движение
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))
        #возврат кортежа порогового изображения вместе с областью, где происходило движение
        return (thresh, (minX, minY, maxX, maxY))
