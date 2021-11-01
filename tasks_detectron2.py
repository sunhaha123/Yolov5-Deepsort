import time
import requests
import redis
conn=redis.Redis(db=1)
# import urllib

import base64
import json
import time
import cv2
from PIL import Image, ImageDraw
import numpy as np
import math
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

# from celery import Celery,platforms
# platforms.C_FORCE_ROOT = True
# backend = 'redis://localhost:6379/1'
# broker = 'redis://localhost:6379/1'
# # 创建celery对象
# app = Celery('deep_protector', backend=backend, broker=broker)

# 得到向量的坐标以及向量的模长
class Point(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def vector(self):
        c = (self.x1 - self.x2, self.y1 - self.y2)
        return c

    def length(self):
        d = math.sqrt(pow((self.x1 - self.x2), 2) + pow((self.y1 - self.y2), 2))
        return d


# 计算向量夹角
class Calculate(object):
    def __init__(self, x, y, m, n):
        self.x = x
        self.y = y
        self.m = m
        self.n = n

    def Vector_multiplication(self):
        self.mu = np.dot(self.x, self.y)
        return self.mu

    def Vector_model(self):
        self.de = self.m * self.n
        return self.de

    def cal(self):
        cos_angle = Calculate.Vector_multiplication(self) / Calculate.Vector_model(self)
        angle = np.arccos(cos_angle)
        angle2 = angle * 360 / 2 / np.pi
        return angle2

def draw_bodys(bodydata,pointsize):
    #    image_origin = Image.open(originfilename)

    body = bodydata
    # right_knee --> right_ankle
    left_shoulder_x = body['left_shoulder_x']
    left_shoulder_y = body['left_shoulder_y']
    left_hip_x = body['left_hip_x']
    left_hip_y = body['left_hip_y']
    right_shoulder_x = body['right_shoulder_x']
    right_shoulder_y = body['right_shoulder_y']
    right_hip_x = body['right_hip_x']
    right_hip_y = body['right_hip_y']


    first_point = Point(left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y)
    second_point = Point(right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y)
    base_point = Point(0, 0, 1, 0)
    ca1 = Calculate(first_point.vector(), base_point.vector(), first_point.length(), base_point.length())  # 左紫
    ca2 = Calculate(second_point.vector(), base_point.vector(), second_point.length(), base_point.length())  # 右橙
    angles = (ca1.cal(), ca2.cal())
    print(angles)
    # 判断角度  0-noraml 1-alarm
    if (angles[0] > 60 and angles[0] < 120) and (angles[1] > 60 and angles[1] < 120):
        alarm_status = 0
    else:
        alarm_status = 1
    # =============================================================================
    # #    画图
    # #    cv2.imshow('image',img)
    # #    cv2.waitKey(0)
    # #    cv2.imwrite(resultfilename,img)
    # #    cv2.destroyAllWindows()
    # =============================================================================
    # 返回绘制后的图片和警报状态
    return  alarm_status, angles

def body_analysis(img_bytes,  pointsize):
    #
    ##    print(filename)
    #    # 二进制方式打开图片文件
    #    f = open(filename, 'rb')
    begin = time.perf_counter()
    request_url = "http://192.168.2.131:5000/body_analysis"
    files = {'image': img_bytes}
    content = requests.post(url=request_url,files=files).text
    end = time.perf_counter()

    print('关节点检测api处理时长:' + '%.2f' % (end - begin) + '秒')
    if content:
        # print(content)
        #        print(content)
        data = json.loads(content)
        # print(data)
        # print(data)
        #是否能从接口返回数据成功
        try:
            result = data
            if result == {}:
                print('NO POINTS!')
                return 0
            else:
                alarm_status, angles = draw_bodys(result, pointsize)
                return alarm_status
        except:
            print('error!')
            return 0


# 创建celery任务
# @app.task(name='number_add')
def number_add(x, y):
    time.sleep(0.25)
    print('number_ %i add 进来了...'%x)
    conn.set('num',x+y)
    # return x + y

# @app.task(name='test_baidu')
def test_bodypoint(person_id,frame):
    # filename =conn.get('photo')
    # result_img,stand_status,angles =  body_analysis('/home/ps/Video_data/test_straight/'+filename,'home/ps/Video_data/test_straight/'+'output'+filename,3)
    print(person_id)
    status = body_analysis(frame, 3)
    print(status)
    conn.rpush('%i' % person_id, status)
    print('num_id: %i' % person_id + '  status is %i' % status)


#每0.5秒苏醒一次 devil
if __name__ =="__main__":

    # p = Pool(4)
    executor = ThreadPoolExecutor(max_workers=6)
    while True:
        time.sleep(0.1)
        #获取需要识别的id
        if conn.scard("alarm_person") ==0:
            continue
        person_id = int(conn.spop('alarm_person').decode())
        #获取对应person_id的图片
        img_bytes = conn.get('photo_%i'%person_id)
        #将图片从bytes转为frame
        task1 = executor.submit(test_bodypoint,person_id,img_bytes)

