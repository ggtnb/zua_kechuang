from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import utime
import image
import aidemo
import random
import gc
import sys
import math
import aicube
from machine import Pin
from machine import FPIOA
from machine import UART
from machine import PWM
import time

# Configure pins
fpioa = FPIOA()
fpioa.set_function(42, FPIOA.PWM0)
fpioa.set_function(52, FPIOA.PWM4)
fpioa.set_function(47,FPIOA.PWM3)
fpioa.set_function(3,FPIOA.UART1_TXD)
fpioa.set_function(32, FPIOA.UART3_TXD)
fpioa.set_function(33, FPIOA.UART3_RXD)
uart = UART(UART.UART3, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
uart_another=UART(UART.UART1,baudrate=115200,bits=UART.EIGHTBITS,parity=UART.PARITY_NONE,stop=UART.STOPBITS_ONE)
servo = PWM(0, 50, 0, enable=True)
servo2 = PWM(4, 50, 0, enable=True)
servo3=PWM(3,50,0,enable=True)

def Servo_360(servo, angle):
    angle = max(-90, min(90, angle))
    duty = 5 + (angle + 90) / 36  # Original duty cycle logic
    servo.duty(int(duty))

# Face Detection Classes
class FaceDetApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.25, nms_threshold=0.3, rgb888p_size=[1920,1080], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.pad(self.get_pad_param(), 0, [104,117,123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]], [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            res = aidemo.face_det_post_process(self.confidence_threshold, self.nms_threshold, self.model_input_size[0], self.anchors, self.rgb888p_size, results)
            if len(res) == 0:
                return res, res
            else:
                return res[0], res[1]

    def get_pad_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * self.rgb888p_size[0])
        new_h = (int)(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return [0,0,0,0,top, bottom, left, right]

class FaceRegistrationApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[1920,1080], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode
        self.umeyama_args_112 = [
            38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041
        ]
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, landm, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]], [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            return results[0][0]

    def svd22(self, a):
        s = [0.0, 0.0]
        u = [0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0]
        s[0] = (math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2) + math.sqrt((a[0] + a[3]) ** 2 + (a[1] - a[2]) ** 2)) / 2
        s[1] = abs(s[0] - math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2))
        v[2] = math.sin((math.atan2(2 * (a[0] * a[1] + a[2] * a[3]), a[0] ** 2 - a[1] ** 2 + a[2] ** 2 - a[3] ** 2)) / 2) if s[0] > s[1] else 0
        v[0] = math.sqrt(1 - v[2] ** 2)
        v[1] = -v[2]
        v[3] = v[0]
        u[0] = -(a[0] * v[0] + a[1] * v[2]) / s[0] if s[0] != 0 else 1
        u[2] = -(a[2] * v[0] + a[3] * v[2]) / s[0] if s[0] != 0 else 0
        u[1] = (a[0] * v[1] + a[1] * v[3]) / s[1] if s[1] != 0 else -u[2]
        u[3] = (a[2] * v[1] + a[3] * v[3]) / s[1] if s[1] != 0 else u[0]
        v[0] = -v[0]
        v[2] = -v[2]
        return u, s, v

    def image_umeyama_112(self, src):
        SRC_NUM = 5
        SRC_DIM = 2
        src_mean = [0.0, 0.0]
        dst_mean = [0.0, 0.0]
        for i in range(0, SRC_NUM * 2, 2):
            src_mean[0] += src[i]
            src_mean[1] += src[i + 1]
            dst_mean[0] += self.umeyama_args_112[i]
            dst_mean[1] += self.umeyama_args_112[i + 1]
        src_mean[0] /= SRC_NUM
        src_mean[1] /= SRC_NUM
        dst_mean[0] /= SRC_NUM
        dst_mean[1] /= SRC_NUM
        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]
        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM
        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])
        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]
        scale = 1.0
        src_demean_mean = [0.0, 0.0]
        src_demean_var = [0.0, 0.0]
        for i in range(SRC_NUM):
            src_demean_mean[0] += src_demean[i][0]
            src_demean_mean[1] += src_demean[i][1]
        src_demean_mean[0] /= SRC_NUM
        src_demean_mean[1] /= SRC_NUM
        for i in range(SRC_NUM):
            src_demean_var[0] += (src_demean_mean[0] - src_demean[i][0]) * (src_demean_mean[0] - src_demean[i][0])
            src_demean_var[1] += (src_demean_mean[1] - src_demean[i][1]) * (src_demean_mean[1] - src_demean[i][1])
        src_demean_var[0] /= SRC_NUM
        src_demean_var[1] /= SRC_NUM
        scale = 1.0 / (src_demean_var[0] + src_demean_var[1]) * (S[0] + S[1])
        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale
        return T

    def get_affine_matrix(self, sparse_points):
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            matrix_dst = self.image_umeyama_112(sparse_points)
            matrix_dst = [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                          matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]
            return matrix_dst

class FaceRecognition:
    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size, reg_input_size, database_dir, anchors, confidence_threshold=0.25, nms_threshold=0.3, face_recognition_threshold=0.75, rgb888p_size=[1280,720], display_size=[1920,1080], debug_mode=0):
        self.face_det_kmodel = face_det_kmodel
        self.face_reg_kmodel = face_reg_kmodel
        self.det_input_size = det_input_size
        self.reg_input_size = reg_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.face_recognition_threshold = face_recognition_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode
        self.max_register_face = 100
        self.feature_num = 128
        self.valid_register_face = 0
        self.db_name = []
        self.db_data = []
        self.face_det = FaceDetApp(self.face_det_kmodel, model_input_size=self.det_input_size, anchors=self.anchors, confidence_threshold=self.confidence_threshold, nms_threshold=self.nms_threshold, rgb888p_size=self.rgb888p_size, display_size=self.display_size, debug_mode=0)
        self.face_reg = FaceRegistrationApp(self.face_reg_kmodel, model_input_size=self.reg_input_size, rgb888p_size=self.rgb888p_size, display_size=self.display_size)
        self.face_det.config_preprocess()
        self.database_init()
        self.last_check_time = utime.ticks_ms()
        self.print_interval = 1000
        self.current_status = "no"

    def run(self, input_np):
        det_boxes, landms = self.face_det.run(input_np)
        recg_res = []
        for landm in landms:
            self.face_reg.config_preprocess(landm)
            feature = self.face_reg.run(input_np)
            res = self.database_search(feature)
            recg_res.append(res)
        return det_boxes, recg_res

    def database_init(self):
        with ScopedTiming("database_init", self.debug_mode > 1):
            db_file_list = os.listdir(self.database_dir)
            for db_file in db_file_list:
                if not db_file.endswith('.bin'):
                    continue
                if self.valid_register_face >= self.max_register_face:
                    break
                valid_index = self.valid_register_face
                full_db_file = self.database_dir + db_file
                with open(full_db_file, 'rb') as f:
                    data = f.read()
                feature = np.frombuffer(data, dtype=np.float)
                self.db_data.append(feature)
                name = db_file.split('.')[0]
                self.db_name.append(name)
                self.valid_register_face += 1

    def database_search(self, feature):
        with ScopedTiming("database_search", self.debug_mode > 1):
            v_id = -1
            v_score_max = 0.0
            feature /= np.linalg.norm(feature)
            for i in range(self.valid_register_face):
                db_feature = self.db_data[i]
                db_feature /= np.linalg.norm(db_feature)
                v_score = np.dot(feature, db_feature)/2 + 0.5
                if v_score > v_score_max:
                    v_score_max = v_score
                    v_id = i
            if v_id == -1:
                self.current_status = "no"
                return 'unknown'
            elif v_score_max < self.face_recognition_threshold:
                self.current_status = "no"
                return 'unknown'
            else:
                if self.db_name[v_id] == "111":
                    self.current_status = "yes"
                else:
                    self.current_status = "no"
                result = 'name: {}, score:{}'.format(self.db_name[v_id], v_score_max)
                return result

    def check_status(self):
        current_time = utime.ticks_ms()
        if utime.ticks_diff(current_time, self.last_check_time) >= self.print_interval:
            self.last_check_time = current_time
            return self.current_status
        return None

    def draw_result(self, pl, dets, recg_results):
        pl.osd_img.clear()
        if dets:
            for i, det in enumerate(dets):
                x1, y1, w, h = map(lambda x: int(round(x, 0)), det[:4])
                x1 = x1 * self.display_size[0]//self.rgb888p_size[0]
                y1 = y1 * self.display_size[1]//self.rgb888p_size[1]
                w = w * self.display_size[0]//self.rgb888p_size[0]
                h = h * self.display_size[1]//self.rgb888p_size[1]
                pl.osd_img.draw_rectangle(x1, y1, w, h, color=(255,0,0,255), thickness=4)
                recg_text = recg_results[i]
                pl.osd_img.draw_string_advanced(x1, y1, 32, recg_text, color=(255,255,0,0))

# Person Detection Class
class PersonDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, labels, anchors, confidence_threshold=0.2, nms_threshold=0.5, nms_option=False, strides=[8,16,32], rgb888p_size=[224,224], display_size=[1920,1080], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.last_boxes = []
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.labels = labels
        self.anchors = anchors
        self.strides = strides
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.nms_option = nms_option
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0,0,0,0,top,bottom,left,right], 0, [0,0,0])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]], [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            dets = aicube.anchorbasedet_post_process(results[0], results[1], results[2], self.model_input_size, self.rgb888p_size, self.strides, len(self.labels), self.confidence_threshold, self.nms_threshold, self.anchors, self.nms_option)
            return dets

    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode >0):
            self.last_boxes = []
            if dets:
                pl.osd_img.clear()
                for det_box in dets:
                    x1, y1, x2, y2 = det_box[2], det_box[3], det_box[4], det_box[5]
                    w = float(x2 - x1) * self.display_size[0] // self.rgb888p_size[0]
                    h = float(y2 - y1) * self.display_size[1] // self.rgb888p_size[1]
                    x1 = int(x1 * self.display_size[0] // self.rgb888p_size[0])
                    y1 = int(y1 * self.display_size[1] // self.rgb888p_size[1])
                    x2 = int(x2 * self.display_size[0] // self.rgb888p_size[0])
                    y2 = int(y2 * self.display_size[1] // self.rgb888p_size[1])
                    self.last_boxes.append((x1,y1,x2,y2))
                    if (h < (0.1*self.display_size[0])):
                        continue
                    if (w < (0.25*self.display_size[0]) and ((x1 < (0.03*self.display_size[0])) or (x2 > (0.97*self.display_size[0])))):
                        continue
                    if (w<(0.15*self.display_size[0]) and ((x1<(0.01*self.display_size[0])) or (x2>(0.99*self.display_size[0])))):

                        continue
                    pl.osd_img.draw_rectangle(x1, y1, int(w), int(h), color=(255,0,255,0), thickness=2)
                    pl.osd_img.draw_string_advanced(x1, y1-50, 32, " " + self.labels[det_box[0]] + " " + str(round(det_box[1],2)), color=(255,0,255,0))
            else:
                pl.osd_img.clear()

    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        input_width = self.rgb888p_size[0]
        input_high = self.rgb888p_size[1]
        ratio_w = dst_w / input_width
        ratio_h = dst_h / input_high
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * input_width)
        new_h = (int)(ratio * input_high)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw - 0.1))
        return top, bottom, left, right

def main():
    # Display mode configuration
    display_mode = "lcd"
    if display_mode == "hdmi":
        display_size = [1920, 1080]
    else:
        display_size = [640, 480]

    # Common configuration
    rgb888p_size = [640, 480]

    # Initialize Pipeline
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create()

    # Initialize Face Recognition
    face_det_kmodel_path = "/sdcard/examples/kmodel/face_detection_320.kmodel"
    face_reg_kmodel_path = "/sdcard/examples/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/examples/utils/prior_data_320.bin"
    database_dir = "/sdcard/examples/utils/db/"
    face_det_input_size = [320, 320]
    face_reg_input_size = [112, 112]
    confidence_threshold = 0.5
    nms_threshold = 0.2
    anchor_len = 4200
    det_dim = 4
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len, det_dim))
    face_recognition_threshold = 0.75

    fr = FaceRecognition(
        face_det_kmodel_path,
        face_reg_kmodel_path,
        det_input_size=face_det_input_size,
        reg_input_size=face_reg_input_size,
        database_dir=database_dir,
        anchors=anchors,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        face_recognition_threshold=face_recognition_threshold,
        rgb888p_size=rgb888p_size,
        display_size=display_size
    )

    # Initialize Person Detection
    person_kmodel_path = "/sdcard/examples/kmodel/person_detect_yolov5n.kmodel"
    person_labels = ["person"]
    person_anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
    person_confidence_threshold = 0.2
    person_nms_threshold = 0.6

    person_det = PersonDetectionApp(
        person_kmodel_path,
        model_input_size=[640,640],
        labels=person_labels,
        anchors=person_anchors,
        confidence_threshold=person_confidence_threshold,
        nms_threshold=person_nms_threshold,
        nms_option=False,
        strides=[8,16,32],
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        debug_mode=0
    )
    person_det.config_preprocess()

    # Main loop
    face_recognition_active = True
    last_time = utime.ticks_ms()
    current_mode = None  # None, 'face', or 'person'

    try:
        while True:
            servo.enable(False)
            servo2.enable(False)
            servo3.enable(False)
            os.exitpoint()

            # Check for UART commands
            cmd = uart.read(1)
            if cmd == b'\xd1':  # Face detection command
                current_mode = 'face'
                face_recognition_active=True
                print("Switching to face detection mode")
            elif cmd == b'\xd2':  # Person detection command
                current_mode = 'person'
                print("Switching to person detection mode")
            elif cmd==b'\xe0':
                uart_another.write(bytes([0x07]))
            elif cmd==b'\xe6':
                uart_another.write(bytes([0xa1]))
            elif cmd==b'\xe1':

                servo3.enable(False)
                servo.enable(True)
                Servo_360(servo,90)
                servo2.enable(False)
                time.sleep(1.6)
                servo.enable(False)
                servo2.enable(True)
                Servo_360(servo2,-90)
                time.sleep(1.8)
            elif cmd==b'\xe2':

                servo2.enable(True)
                Servo_360(servo2,90)
                time.sleep(2)
                servo2.enable(False)
                servo.enable(True)
                Servo_360(servo,-90)
                time.sleep(1.9)
                servo.enable(False)

            elif cmd==b'\xe3':

                servo3.enable(True)

                Servo_360(servo3,90)
                time.sleep(1)
                servo3.enable(False)

            elif cmd==b'\xe4':

                servo.enable(False)
                servo2.enable(False)
                servo3.enable (True)
                Servo_360(servo3,-90)
                time.sleep(2)
            elif cmd==b'\x06':
                uart_another.write(bytes([0xff]))
            img = pl.get_frame()

            if current_mode == 'face':
                # Run face detection and recognition
                det_boxes, recg_res = fr.run(img)
                fr.draw_result(pl, det_boxes, recg_res)

                # Check status and send UART commands if needed
                status = fr.check_status()
                if status == 'yes':
                    uart.write(bytes([0xAA, 0x00, 0x01]))
                    uart_another.write(bytes([0x12])
                    current_mode=None

                elif status == 'no':
                    uart.write(bytes([0xAA, 0x00, 0x02]))
                    uart_another.write(bytes([0x11]))
                    time.sleep(3)

            elif current_mode == 'person':
                # Run person detection
                res = person_det.run(img)
                person_det.draw_result(pl, res)



            # Send person detection results via UART
                current_time = utime.ticks_ms()
                if current_time - last_time >= 100:
                    if not person_det.last_boxes:
                        uart_another.write(bytes([0xff]))
                    else:
                        for box in person_det.last_boxes:
                            x1, y1, x2, y2 = box
                            x = (x1 + x2) / 2
                            y = (y1 + y2) / 2
                            print("{},{},{},{}                        ".format(x1, x2, y1, y2))
                            if x > 210 and x < 470:
                                uart_another.write(bytes([0x02]))
                            if x < 210:
                                uart_another.write(bytes([0x04]))
                            if x > 470:
                                uart_another.write(bytes([0x05]))


            pl.show_image()
            gc.collect()

    finally:
        # Cleanup0
        if 'fr' in locals():
            del fr
        if 'person_det' in locals():
            person_det.deinit()
        pl.destroy()

if __name__ == "__main__":
    main()
