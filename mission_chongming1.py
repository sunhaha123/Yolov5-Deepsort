import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from AIDetector_pytorch import Detector
import imutils
import numpy as np
import cv2
import time
from flask import Flask, render_template, Response,request, jsonify
import pandas as pd
import redis
import threading
app = Flask(__name__)
conn1=redis.Redis(db=1)
#清空redis 0数据库
for elem in conn1.keys():
    conn1.delete(elem)
conn1.set('lane', 0)
conn1.set('zone',0)
conn1.set('person_id', 0)

conn2=redis.Redis(db=2)
#清空redis 1数据库
for elem in conn2.keys():
    conn2.delete(elem)
conn2.set('lane', 0)
conn2.set('zone',0)
conn2.set('person_id', 0)

conn3=redis.Redis(db=3)
#清空redis 1数据库
for elem in conn3.keys():
    conn3.delete(elem)
conn3.set('lane', 0)
conn3.set('zone',0)
conn3.set('person_id', 0)

qmap={}

import requests
import base64
import json
from io import BufferedReader, BytesIO
import io
from concurrent.futures import ThreadPoolExecutor

from  diff_lane import distance,beyond
from trackId import  update_tracker
#初始化detctor
from  AIDetector_pytorch import Detector
det = Detector()

#初始化deepsort
import trackId
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
tracker1 = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
tracker2 = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

tracker3 = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
#
columns = ['id', 'person_id', 'timestamp', 'status', 'lane', 'section', 'central_x', 'central_y', 'width', 'height']
def discriminantStatic(judge_person,width,height):
    title = 1
    diff_value = judge_person.groupby(['person_id'])['central_x'].max() - \
                 judge_person.groupby(['person_id'])['central_x'].min()
    mean_value_x = judge_person.groupby(['person_id'])['central_x'].mean().item()
    mean_value_y = judge_person.groupby(['person_id'])['central_y'].mean().item()
    if diff_value.item() < 0.04:
        title = 0
        if distance(mean_value_x,mean_value_y,width,height) == -1:
              title = -1
    return title

def discriminantDynamic(judge_person):
    title = 1
    width_value_x_min = judge_person.groupby(['person_id'])['width'].min()
    width_value_y_min = judge_person.groupby(['person_id'])['height'].min()
    width_value_x_max = judge_person.groupby(['person_id'])['width'].max()
    width_value_y_max = judge_person.groupby(['person_id'])['height'].max()
    squere_min = width_value_x_min * width_value_y_min
    squere_max = width_value_x_max * width_value_y_max
    if squere_max.item() - squere_min.item() > 10:
        title = 0
    return title

def protect(slide_box,magic_box,frame,width,height,conn):
    # ---begin protecter-----
    tp = time.time()
    t0=time.time()
    columns = ['id', 'person_id', 'timestamp', 'status', 'lane', 'section', 'central_x', 'central_y', 'width', 'height']
    temp_box = pd.DataFrame(magic_box, columns=columns)
    print('df wasted time %5f' % (time.time() - t0))
    t0 = time.time()
    slide_box = pd.concat([temp_box, slide_box], axis=0)
    print('concat wasted time %5f' % (time.time() - t0))
    # =============================================================================
    #      去除过期的person_id  超过30s
    # =============================================================================
    t0 = time.time()
    if len(slide_box)!=0:
        slide_box['timestamp'] = np.where(t0-slide_box['timestamp']/1000<20,slide_box['timestamp'],np.nan)
        slide_box.dropna(subset=['timestamp'],inplace=True)
        print('delete wasted time %5f'%(time.time()-t0))
    # =============================================================================
    #     需要判别是否溺水的人
    # =============================================================================
    t0 = time.time()
    consist_person = slide_box.groupby(['person_id'])['id'].count()
    need_cal_person = []
    for consist in consist_person:
        if consist > 15*4 :
            need_cal_person.append(consist_person[consist_person.values == consist].index[0])
    print('need_cal watestd time %5f' % (time.time() - t0))
    # =============================================================================
    #     溺水判别 最终得到alarm_num
    # =============================================================================
    t0 = time.time()
    alarm_num = []
    attention_num = []
    need_cal_person = list(set(need_cal_person))
    for j in range(len(need_cal_person)):
        tt = time.time()
        judge_person = slide_box[slide_box['person_id'] == need_cal_person[j]]

        if judge_person.iloc[0,2]<judge_person.iloc[1,2]:
            judge_person = judge_person.sort_values(by=['timestamp'], ascending=False)
        print('isinsort wasted time %5f' % (time.time() - tt))
        judge_person = judge_person.reset_index(drop=True)
        print('sort wasted time %5f'%(time.time()-t0))
        # 动、静态溺水判别
        t1 = time.time()
        if np.abs(judge_person.iloc[0,6]-judge_person.iloc[-1,6])<0.06:
            if distance(judge_person.iloc[0,6],judge_person.iloc[0,7],width,height) == -1:
                recognition =-1
            else:
                recognition = 1
        else:
            recognition =0
        # recognition = discriminantStatic(judge_person,width,height)
        print('discriminant wasted time %5f'%(time.time()-t1))
        if recognition == 1:  # 为0 alarm
            ## 推送person_Id和框图进行骨骼关键点识别和姿态识别
            idx = judge_person.loc[0,'person_id']
            lefttop_x = judge_person.loc[0,'central_x'] - 0.5 * judge_person.loc[0,'width']
            lefttop_y = judge_person.loc[0,'central_y'] - 0.5 * judge_person.loc[0,'height']
            rightbottom_x = judge_person.loc[0,'central_x'] + 0.5 * judge_person.loc[0,'width']
            rightbottom_y = judge_person.loc[0,'central_y'] + 0.5 * judge_person.loc[0,'height']
            alarm_imagex = frame[int(lefttop_y * height):int(rightbottom_y * height),
                          int(lefttop_x * width):int(rightbottom_x * width)]
            #* 使用celery
            # result1 = test_baidu.delay(idx ,alarm_imagex)
            #* 使用redis
            t3=time.time()
            ret, img_encode = cv2.imencode('.jpg', alarm_imagex)
            alarm_bytes = img_encode.tobytes()
            # 将最新的报警图片存入redis
            conn.set('photo_%i' % idx, alarm_bytes)
            #将需要检测的泳客id 放入set
            conn.sadd('alarm_person',idx)
            print('imencode wasted time %5f'%(time.time()-t3))
            #**从redis中获取泳客通过姿态识别判别后的溺水状态 0-正常 1-溺水
            status_list = conn.lrange('%i' %idx,-10,-1)
            if len(status_list)>5:
                sum=0
                for each in status_list:
                    each = int(each.decode())
                    sum+=each
                average=float(sum/len(status_list))
                if average>0.5:
                    alarm_num.append(need_cal_person[j])
        if recognition == -1:
            attention_num.append(need_cal_person[j])
            # slide_box.loc[slide_box['person_id'].isin(alarm_num), 'status']
        # print('%d dis wasted time %5f'%(1j,time.time()-t1))
    print('alarm_cal watestd time %5f' % (time.time() - t0))
    # =============================================================================
    #          画线
    # =============================================================================
    t0 = time.time()
    result_frame  = frame
    count = 0
    people_num = 0
    message =0
    for item in magic_box:
        # 根据magic_box输出及时调整
        # '0-id','1-person_id','2-timestamp',3-status','4-lane',
        # '5-section','6-central_x','7-central_y','8-w','9-h'

        person_id = item[1]
        central_x = item[6]
        central_y = item[7]
        lefttop_x = central_x - 0.5 * item[8]
        lefttop_y = central_y - 0.5 * item[9]
        rightbottom_x = central_x + 0.5 * item[8]
        rightbottom_y = central_y + 0.5 * item[9]
        lane = item[4]
        zone = item[5]
        # print(person_id)

        #         slide_box=np.vstack([slide_box,item])
        #
        people_num += 1
        if person_id in attention_num:
            # cv2.putText(result_frame, str(person_id), (int(lefttop_x * width), int(lefttop_y * height)), 0,
            #             1, (255, 255, 255), 2)
            # cv2.putText(result_frame, 'alarm',
            #             (int((lefttop_x + (centrol_x - lefttop_x) * 2) * width - 80),
            #              int(lefttop_y * height)), 0, 1, (0, 255, 255), 2)
            cv2.rectangle(result_frame, (int(lefttop_x * width), int(lefttop_y * height)),
                          (int(rightbottom_x * width), int(rightbottom_y * height)), (0, 255, 255), 1)
            magic_box[count][3] = -1
        else:
            if person_id in alarm_num:
                # cv2.putText(result_frame, str(person_id), (int(lefttop_x * width), int(lefttop_y * height)), 0,
                #             1, (255, 255, 255), 2)
                # cv2.putText(result_frame, 'alarm',
                #             (int((lefttop_x + (central_x - lefttop_x) * 2) * width - 80),
                #              int(lefttop_y * height)), 0, 0.5, (0, 0, 255), 1)
                cv2.rectangle(result_frame, (int(lefttop_x * width), int(lefttop_y * height)),
                              (int(rightbottom_x * width), int(rightbottom_y * height)), (0, 0, 255), 2)

                alarm_image = result_frame[int(lefttop_y * height):int(rightbottom_y * height),
                                  int(lefttop_x * width):int(rightbottom_x * width)]
                #保存alarm
                ret, img_encode = cv2.imencode('.jpg', alarm_image)
                str_encode = img_encode.tostring()  # 将array转化为二进制类型
                f4 = BytesIO(str_encode)  # 转化为_io.BytesIO类型
                f4.name = '....jpg'  # 名称赋值
                f5 = BufferedReader(f4)
                files = {'alarmphoto': f5}
                reponse_txt = requests.post('http://127.0.0.1:8011/camera/alarmphoto',files=files).text
                # reponse_txt = requests.post('http://222.70.180.158:8010/camera/alarmphoto', files=files).text
                print(reponse_txt)
                # 保存alarmEvent
                r_lane = int(conn.get('lane').decode())
                r_zone =int(conn.get('zone').decode())
                r_person_id = int(conn.get('person_id').decode())
                if r_lane!=int(lane) or r_zone!=int(zone) or r_person_id!=int(person_id):
                    success, encoded_image = cv2.imencode(".jpg", result_frame)
                    io2 = BytesIO(encoded_image.tobytes())
                    io2.name = '.jpg'
                    io2 = BufferedReader(io2)
                    #alarmimage 生成
                    ret, img_encode = cv2.imencode('.jpg', alarm_image)
                    str_encode = img_encode.tostring()
                    f4 = BytesIO(str_encode)  # 转化为_io.BytesIO类型
                    f4.name = '....jpg'  # 名称赋值
                    f5 = BufferedReader(f4)
                    files = {'alarmphoto': f5,'eventphoto':io2}
                    #写入缓存保存对象
                    conn.set('lane', lane)
                    conn.set('zone', zone)
                    conn.set('person_id', person_id)
                    reponse_txt = requests.post('http://127.0.0.1:8011/camera/alarmeventphoto', files=files).text
                    # reponse_txt = requests.post('http://222.70.180.158:8010/camera/alarmeventphoto', files=files).text
                    print('Event'+reponse_txt)

            ####################################
                magic_box[count][3] = 0  # 0为溺水状态
            else:

                pass
        count += 1

    # 输出非正常泳道的泳客信息
    if alarm_num != []:
        for item in magic_box:
            if item[3] == 0:
                message = item[4]
                print('message:%s'%str(message))
                break
            else:
                message = '0'
    else:
        message = '0'  # '各泳道一切正常！'

    # 转换frame为bytes
    t0=time.time()
    ret, jpeg = cv2.imencode('.bmp', frame)
    print('imencode wasted time %5f' % (time.time() - t0))
    bframe = jpeg.tobytes()
    conn.set('people', int(people_num))
    conn.set("camera_status", int(message))
    conn.set("video", b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + bframe + b'\r\n\r\n')
    print('protect wasted time %5f'%(time.time()-tp))
    return slide_box


def hardwork(name,det,tracker,url,conn,line1,line2,screenshot):

    # name = 'demo'
    # url = 'rtsp://admin:xike123456@fanmaoyang.kmdns.net:58084/stream1'
    # url ='/home/ps/Video_data/hongkou/alarm_test.mp4'
    cap = cv2.VideoCapture(url)
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    videoWriter = None
    # 初始化slide_box
    columns = ['id', 'person_id', 'timestamp', 'status', 'lane', 'section', 'central_x', 'central_y', 'width', 'height']
    slide_box = pd.DataFrame(columns=columns)

    while True:
        t0= time.time()
        # try:
        flag, frame = cap.read()
        if flag != True:
            print('%s video  is  break!  '%name )
            st = time.time()
            cap = cv2.VideoCapture(url)
            print('%s total time lost due to reinititation :%d s' %(name,time.time() - st))
            continue
        #画线  截图
        spot1,spot2=line1[0],line1[1]
        spot3, spot4 = line2[0], line2[1]
        shot1,shot2,shot3,shot4 = screenshot[0],screenshot[1],screenshot[2],screenshot[3]
        cv2.line(frame, spot1, spot2, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, spot3, spot4,(0, 0, 255), 2, lineType=cv2.LINE_AA)
        frame = frame[shot1:shot2, shot3:shot4]

        #识别、追踪
        t0=time.time()
        try:
            _, bboxes = det.detect(frame)
            print('%s detect wasted time %5f' % (name,time.time() - t0))
        except:
            print('%s detect error!'%name)
            frame =frame

        #追踪
        if len(bboxes) > 0:
            result = update_tracker(tracker,bboxes, frame)
            result_frame = result[0]
            magic_box = result[1]
        else:
            result_frame = frame
            magic_box = []

        height,width = result_frame.shape[0],result_frame.shape[1]
        #保存视频
        # if videoWriter is None:
            # fourcc = cv2.VideoWriter_fourcc(
            #     'm', 'p', '4', 'v')  # opencv3.0
            # videoWriter = cv2.VideoWriter(
            #     'result1.mp4', fourcc, fps, (result_frame.shape[1], result_frame.shape[0]))


        slide_box = protect(slide_box,magic_box,result_frame,width,height,conn)
        #############测试延迟#################
        time.sleep(0.20)
        ####################################
        print('%s total wasted time %5f'%(name,time.time()-t0))

def main():
    #开启flask
    t2 = threading.Thread(target=start_web_service, name='10001', args=())
    t2.start()
    #开启线程进行识别
    # url1 = r'rtsp://admin:xike123456@192.168.7.100:554/ch1/main/av_stream'
    # url1 = r'/home/ps/Video_data/chongming/chongming_test.mp4'
    url1 = r'rtsp://admin:xike123456@fanmaoyang.kmdns.net:58090/stream1'
    # url1 = r'/home/ps/Video_data/hongkou/alarm_test.mp4'
    qmap['1'] = conn1
    line1 = ((763,680),(2154,680))
    line2 = ((763,680),(294,1132))
    # screenshot = [402,895, 97,1920]
    # screenshot  = [676,1380, 306,2560] #2560*1440
    go1 = threading.Thread(target=hardwork, name='stream1', args=('work1',det,tracker1,url1,conn1,line1,line2,screenshot))
    go1.start()

    # line1 = ((763, 680), (2154, 680))
    # line2 = ((763, 680), (294, 1132))
    # screenshot  = [676,1380, 306,2560]
    # url2 = r'rtsp://admin:xike123456@fanmaoyang.kmdns.net:58084/stream1'
    # qmap['2'] = conn2
    # go2 = threading.Thread(target=hardwork, name='stream2', args=('work2',det,tracker2,url2,conn2,line1,line2,screenshot))
    # go2.start()
    #
    # line1 = ((763, 680), (2154, 680))
    # line2 = ((763, 680), (294, 1132))
    # screenshot = [676, 1380, 306, 2560]
    # url3 = r'rtsp://admin:xike123456@fanmaoyang.kmdns.net:58084/stream1'
    # qmap['3'] = conn2
    # go3 = threading.Thread(target=hardwork, name='stream3',
    #                        args=('work3', det, tracker3, url3, conn3, line1, line2, screenshot))
    # go3.start()


def test():
    while True:
        msg = str(conn1.blpop('photo')[1].decode())
        output=  test_baidu(msg)
        if output:
            # pool_output = p.map(test_baidu, msg)
            print(output)
        else:
            print('waiting!')

def gen(camid):
    conn = qmap.get(camid)
    while True:
        frame = conn.get('video')
        yield (frame)



@app.route('/video_feed')
def video_feed():
    cam_id = request.args.get("cam_id")
    return Response(gen(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')


def start_web_service():

    app.run(host='0.0.0.0', port=8001)

# def start_web_service2():
#     app.run(host='0.0.0.0', port=8002)

if __name__ == '__main__':
    # t2 = threading.Thread(target=start_web_service, name='10001', args=())
    # t3 = threading.Thread(target=start_web_service2, name='10002', args=())
    # t2.start()
    # t3.start()
    main()



