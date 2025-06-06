#!/usr/bin/python3.8
# System libraries
import cv2
import numpy as np
import torch
import time
# Object detection library

# Zed-Mini camera library
# from gate import *
from rov_move_functions import *
from functions import *
from rov import *
import threading

import argparse
import signal

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

class Torpedo:
    def __init__(self):
        self.model                      = None
        self.frame                      = None
        self.angle                      = None
        self.result                     = None
        self.out                        = None

        self.functions                  = Functions()
        self.rov                        = Rov('ttyUSB0')
        self.commands                   = RovMoveCommands(self.rov)

        self.center                     = [0, 0]
        self.v_mid_time                 = 5
        self.h_mid_time                 = 2
        self.max_depth                  = 0.2
        self.min_depth                  = 0.0
        self.wait_right_left            = 8
        self.tolerance                  = 5
        self.tolerance_diff_x           = 25.0
        self.ray_data                   = 0
        self.left_count                 = 0
        self.search_step                = 0
        self.step                       = 0
        self.area                       = 0
        self.frame_count                = 0
        self.count_destroy              = 0
        self.fark                       = 0
        self.a                          = 0
        self.atis                       = 0
        self.adet                       = 0

        self.turn_ok                    = True
        self.left_turn_is_finished      = True
        self.right_turn_is_finished     = True
        self.yok_et                     = False
        self.yolo_ok                    = False
        self.left                       = False
        self.right                      = False
        self.ortala                     = False
        self.destroy                    = False
        self.fark_y                     = 0

        self.start_time                 = time.time() + 7 #7 second delay added because code starts 7 second later 
        self.left_wait_time             = time.time()
        self.right_wait_time            = time.time()

        self.class_list                 = self.load_classes()
        self.colors                     = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
        self.is_cuda                    = len(sys.argv) > 1 and sys.argv[1] == "cuda"
        self.net                        = self.build_model(self.is_cuda)

    def handler(self, signum, frame):
        res = input("Ctrl-c was pressed. Do you really want to exit? y/n ")

        if res == 'y':
            self.commands.values.pwm_turn = 0
            self.commands.values.pwm_forward = 0
            self.commands.values.pwm_left = 0
            self.commands.values.pwm_right = 0

            self.commands.moveForward()
            self.commands.moveTurn()
            self.commands.moveLeft()
            self.commands.moveRight()
            self.rov.run()
            exit(1)

    def build_model(self, is_cuda=True):
        self.net = cv2.dnn.readNet("/home/creatiny/underwater/models/cember_15_07_m.onnx")
        if True:
            print("Attempty to use CUDA")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return self.net

    def detect(self, image, net):
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        return preds

    def load_classes(self):
        class_list = []
        with open("/home/creatiny/yolov5-opencv-cpp-python/config_files/classes1.txt", "r") as f:
            class_list = [cname.strip() for cname in f.readlines()]
        return class_list

    def wrap_detection(self, input_image, output_data):
        class_ids = []
        confidences = []
        boxes = []
        rows = output_data.shape[0]
        image_width, image_height, _ = input_image.shape
        x_factor = image_width / INPUT_WIDTH
        y_factor =  image_height / INPUT_HEIGHT

        for r in range(rows):
            row = output_data[r]
            confidence = row[4]
            if confidence >= 0.7:
                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > 0.6 and class_id == 1): #Hangi renk cember tespiti gerçekleştirilecekse class_id ye esitlenmesi lazım. 0 = kirmizi, 1 = yesil, 2 = mavi
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left =(x - 0.5 * w) * x_factor
                    left = int(left)
                    top = ((y - 0.5 * h) * y_factor)
                    top = int(top)
                    width = (w * x_factor)
                    width = int(width)
                    height = (h * y_factor)
                    height = int(height)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 
        result_class_ids = []
        result_confidences = []
        result_boxes = []
        for i in indexes:
            result_confidences.append(confidences[i])
            result_class_ids.append(class_ids[i])
            result_boxes.append(boxes[i])

        return result_class_ids, result_confidences, result_boxes

    def search(self):
        self.curr_time = time.time() - self.start_time
        text = ""
        if self.search_step == 0:
            if self.step == 0:
                if self.curr_time < self.v_mid_time:   #10 saniye geçiyor
                    self.commands.values.pwm_forward = 155
                    self.commands.moveForward()
                    text = "vertical mid icin ilerle"
                else:
                    # self.commands.values.pwm_turn = -90
                    # self.commands.moveTurn()

                    self.start_time = time.time()
                    self.step       = 1
                    text = "horizontal mid icin don"
            else:
                if self.curr_time < self.h_mid_time:   #4 saniye geçti
                    self.commands.values.pwm_forward = 151
                    self.commands.moveForward()
                    text = "horizontal mid icin ilerle"
                else:
                    self.start_time  = time.time()
                    self.search_step = 1
                    text = "horizontal mid icin dur"

        else:
            if self.curr_time < 5:
                self.commands.values.pwm_forward = 154
                self.commands.moveForward()
                text = "duz git 5sn"
            elif self.curr_time >= 5 and self.left_count == 0:
                self.left        = True
                self.left_count += 1

            if  self.left == True and self.left_turn_is_finished == True and self.curr_time >= 5:
                self.commands.values.pwm_turn = -90
                self.commands.moveTurn()

                self.left_turn_is_finished = False
                self.left_wait_time        = time.time()

            if self.left == True and self.left_turn_is_finished == False:

                self.commands.values.pwm_forward = 0
                self.commands.moveForward()

                if time.time() - self.left_wait_time > self.wait_right_left and self.curr_time >= 5:
                    self.left_turn_is_finished = True
                    self.right                 = True
                    self.left                  = False

            if self.right == True and self.right_turn_is_finished == True:

                self.commands.values.pwm_turn = 180
                self.commands.moveTurn()

                self.right_turn_is_finished = False
                self.right_wait_time        = time.time()

            if self.right == True and self.right_turn_is_finished == False:
                self.commands.values.pwm_forward = 0
                self.commands.moveForward()

                if time.time() - self.right_wait_time > self.wait_right_left and self.curr_time >= 5:
                    self.right_turn_is_finished = True
                    self.ortala                 = True
                    self.right                  = False

            if self.ortala == True and (self.left_turn_is_finished == True and self.right_turn_is_finished == True) :
                self.commands.values.pwm_turn = -90
                self.commands.moveTurn()

                self.left_count = 0

                self.left_turn_is_finished  = True
                self.right_turn_is_finished = True
                self.ortala                 = False

                self.start_time = time.time()

        cv2.putText(self.frame, text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"self.curr_: {self.curr_time}"             , (300,420), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.left_count: {self.left_count}"  , (300,360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.right     : {self.right}"  , (300,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.right_is: {self.right_turn_is_finished}"  , (300,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.left: {self.left}"  , (300,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.left_is: {self.left_turn_is_finished}"  , (300,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.ortala: {self.ortala}"  , (300,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.step: {self.step}"  , (300,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        # cv2.putText(self.frame,f"self.search_step: {self.search_step}"  , (300,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)

    # is_the_vehicle_centered
    def is_the_vehicle_centered(self):
        if self.angle > -1 * (self.tolerance) and self.angle < self.tolerance:
            return True
        else:
            return False

    def is_the_vehicle_centered_diff_x(self):
        if self.fark >= -1 * (self.tolerance_diff_x) and self.fark <= (self.tolerance_diff_x):
            return True
        else:
            return False

    def is_the_vehicle_centered_diff_y(self):
        if self.fark_y >= -1 * (self.tolerance_diff_x) and self.fark_y <= (self.tolerance_diff_x):
            return True
        else:
            return False

    def ray_check(self):
        self.ray_data = self.rov.receivedData.battV
        if self.ray_data < 70 and  self.ray_data > 0:
            pass

    def basinc(self):
        if self.rov.receivedData.depth < self.min_depth or self.rov.receivedData.depth > self.max_depth:
            if self.rov.receivedData.depth > self.max_depth:
               self.commands.values.pwm_up = 120
               self.commands.moveUp()

            if self.rov.receivedData.depth < self.min_depth:
               self.commands.values.pwm_down = 150  
               self.commands.moveDown()
        else:
            self.commands.values.pwm_down = 0
            self.commands.values.pwm_up   = 0
            self.commands.moveUp()

        cv2.putText(self.frame,f"depth: {self.rov.receivedData.depth:.2f}", (10,330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)

    # run_yolo
    def run_yolo(self):
        inputImage = self.frame
        outs = self.detect(inputImage, self.net)

        class_ids, confidences, boxes = self.wrap_detection(inputImage, outs[0])
        if len(boxes) > 0:
            for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                    color = (0, 0, 0)
                    cv2.rectangle(self.frame, box, color, 3)
                    self.area = box[2] * box[3]
                    self.center = [(box[0] + box[0] + box[2])/ 2, 0]
                    self.yolo_ok = True

                    if 210000 > self.area > 500 and self.count_destroy == 0:
                        self.destroy = True
                        self.count_destroy += 1
        else:
            self.yolo_ok = False
            self.center = [0, 0]
            self.area   = 0

    # Callback
    def callback(self):
        self.run_yolo()

        # self.basinc()
        # self.rov.run()

        if self.yok_et == True: #Icinden gececek

            if time.time() - self.start_time > 5:
                self.rov.commandData.lightControl = 0.0
                self.start_time = time.time()
            elif time.time() - self.start_time > 4:
                self.rov.commandData.lightControl = 1.0
            else:
                self.commands.values.pwm_down = 106
                self.commands.moveDown()
            # cy = self.center[1]
            # self.fark_y = int(370 - cy)

            # if self.is_the_vehicle_centered_diff_y() != True:
            #     if self.fark_y > 0:
            #         self.commands.values.pwm_down = 106
            #         self.commands.values.pwm_forward = 0
            #         self.commands.moveForward()
            #         self.commands.moveDown()
            #     if self.fark_y < 0 :
            #         self.commands.values.pwm_up = 105
            #         self.commands.values.pwm_forward = 0
            #         self.commands.moveForward()
            #         self.commands.moveUp()
            # else:
            #     self.commands.values.pwm_up = 0
            #     self.commands.moveUp()
            #     self.a = 1

            # if self.a == 1 or self.fark_y == 340:
            #     self.commands.values.pwm_up = 0
            #     self.commands.values.pwm_forward = 0
            #     self.commands.moveForward()
            #     self.commands.moveUp()

            #     self.rov.commandData.lightControl = 1.0
            self.rov.run()

        elif self.destroy == True and self.yolo_ok == True: #Aramayi birakacak ortalayarak yaklasacak
            cx = self.center[0]
            self.fark = int(320 - cx)
            self.angle = self.functions.calculateAngle(cx)

            if self.is_the_vehicle_centered_diff_x() != True:
                if self.fark < 0:
                    self.commands.values.pwm_right = 106
                    self.commands.moveRight()
                if self.fark > 0 :
                    self.commands.values.pwm_left = 105
                    self.commands.moveLeft()
            else:
                self.turn_ok = True
                self.commands.values.pwm_right = 1
                self.commands.values.pwm_left = 2
                self.commands.moveRight()
                self.commands.moveLeft()

                self.commands.values.pwm_forward = 98
                self.commands.moveForward()
            self.rov.run()
            cv2.putText(self.frame,f"vehicle_centered: {self.is_the_vehicle_centered_diff_x()}"  , (300,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)

            if self.area > 210000: #Yeterince yaklasinca ortalamayi birakacak dumduz gidecek
                self.yok_et     = True
                self.destroy    = False
                self.start_time = time.time()

        elif self.destroy == True and self.yolo_ok == False:
            self.commands.values.pwm_forward = 51
            self.commands.moveForward()

        else:
            if self.yolo_ok == True:
                cx = self.center[0]
                self.fark = int(320 - cx)
                self.angle = self.functions.calculateAngle(cx)

                if self.is_the_vehicle_centered_diff_x() != True:
                    if self.fark < 0:
                        self.commands.values.pwm_right = 106
                        self.commands.moveRight()
                    if self.fark > 0 :
                        self.commands.values.pwm_left = 105
                        self.commands.moveLeft()
                else:
                    self.turn_ok = True
                    self.commands.values.pwm_right = 1
                    self.commands.values.pwm_left = 2
                    self.commands.moveRight()
                    self.commands.moveLeft()

                    self.commands.values.pwm_forward = 98
                    self.commands.moveForward()
                self.rov.run()
            else:
                self.turn_ok = True
                self.search()
            self.rov.run()
            


        self.ray_check()
        self.rov.run()
        


        cv2.putText(self.frame,f"self.angle: {self.angle}"  , (300,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"self.fark: {self.fark}"  , (300,320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"self.area: {self.area}"  , (300,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"self.destroy: {self.destroy}"  , (300,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"self.yolo_ok: {self.yolo_ok}"  , (300,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"self.yok_et: {self.yok_et}"             , (300,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"self.a: {self.a}"             , (300,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"fark_y: {self.fark_y}"             , (300,500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(self.frame,f"ileri: {self.commands.values.pwm_forward}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"turn : {self.commands.values.pwm_turn}"   , (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"left : {self.commands.values.pwm_left}"   , (10,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"right: {self.commands.values.pwm_right}"  , (10,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"up   : {self.commands.values.pwm_up}"     , (10,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(self.frame,f"down : {self.commands.values.pwm_down}"   , (10,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)

        cv2.putText(self.frame,f"yaw: {self.rov.receivedData.yaw}"  , (300,280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)    

        # cv2.imshow("video",self.frame)
        # cv2.waitKey(25)

    # start
    def start(self, model):
        #camera = cv2.VideoCapture("/home/creatiny/underwater/video/03.08/torpido_havuz_10.avi")
        camera = cv2.VideoCapture(0,cv2.CAP_V4L2)
        if not camera.isOpened():
            print("Kamera açilamadi")
            exit()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'mp4v' codec kullanılır
        out    = cv2.VideoWriter('/home/creatiny/underwater/video/14.08/torpido_havuz_15.avi', fourcc, 10.0, (640, 640))
        out2   = cv2.VideoWriter('/home/creatiny/underwater/video/14.08/torpido_havuz_16.avi', fourcc, 10.0, (640, 640))

        self.model = model
        signal.signal(signal.SIGINT, self.handler)
        start_time = time.time()
        while True:
            _, frame = camera.read()
            self.frame_count += 1

            if _ == True:
                frame = cv2.resize(frame, (640,640)) # Model imgsz'i 640 verdğimiz icin resize yaparak goruntuyyu 640x640 yapmamız gerekiyor. Yoksa tespit eder fakat hatali kordinat verir. 
                self.frame = frame
                out2.write(self.frame)

                self.callback()
                # self.commands.values.pwm_down = 50
                # self.commands.moveDown()
                # self.rov.run()
                # self.commands.values.pwm_forward = 250
                # self.commands.moveForward()
                # self.rov.run()
                # self.commands.values.pwm_right = 250
                # self.commands.moveRight()
                # self.rov.run()

                # if time.time() - self.start_time > 30:
                #     self.rov.commandData.lightControl = 1.0
                # else:
                #     self.rov.commandData.lightControl = 0.0
                
                # print(time.time() - self.start_time)

                # elapsed_time = time.time() - start_time
                # if elapsed_time >= 1.0:
                #     fps = self.frame_count / elapsed_time
                #     start_time = time.time()
                #     self.frame_count = 0
                # cv2.putText(self.frame,f"fps: {fps}"  , (300,600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), thickness=1, lineType=cv2.LINE_AA)

            out.write(self.frame)

        camera.release()
        out.release()
        cv2.destroyAllWindows()
