import cv2 as cv
import runpy as np

import os

class Haar:
    def __init__(self, 
                 path_to_cascade: str):
        path_to_cascade = 'data/haarcascade_frontalface_alt2.xml'

        self.classifier = cv.CascadeClassifier(path_to_cascade)

    
    def detect (self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = self.classifier.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            color = (0,0,255)
            width = w + x
            height = h + y
            cv.rectangle(frame, (x, y), (width, height), color, 2)


class DNN:
    def __init__(self,
                 model_file: str,
                 config_file: str,
                 conf_threshold = 0.8):

        self.conf_threshold = conf_threshold
        self.net = cv.dnn.readNetFromCaffe(config_file, model_file)


    def __get_human_from_frame (self, frame, detections):
        # to do:
        # сделать нормальную вырезку лица с фото
        # сделать алгоритм по которому будет извлекаться фото из потока и сохраняться на диск
    
        frame_height = len(frame)
        frame_width = len(frame[0])

        x1 = int(detections[3] * frame_width)
        y1 = int(detections[4] * frame_height)
        x2 = int(detections[5] * frame_width)
        y2 = int(detections[6] * frame_height)

        if cv.waitKey(1) & 0xFF == ord('s'):
            new_frame = frame[y1-50:frame_width, x1-100:x2+100]
            cv.imwrite("viktor.jpg", new_frame)

        cv.rectangle(frame, (x1,y1), (x2,y2), color=(0,255,0), thickness=3)


    def detect(self, frame):
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

        self.net.setInput(blob)
        network_output = self.net.forward()

        for detection in network_output[0, 0]:
            confidence = detection[2]
            
            if confidence > self.conf_threshold:
                self.__get_human_from_frame (frame, detection)
               

def run():
    cap = cv.VideoCapture(1)

    modelFile = "caffe_model/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "caffe_model/deploy.prototxt"

    dnn_detector = DNN (model_file=modelFile, config_file=configFile)

    while (True):    
        ret, frame = cap.read()

        if ret == True:            
            dnn_detector.detect(frame)

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    run()
