from AIDetector_pytorch import Detector
import imutils
import cv2
import time

def main():

    name = 'demo'

    det = Detector()
    # cap = cv2.VideoCapture('/home/ps/Video_data/chongming/0723/chongming_0723.avi')
    cap = cv2.VideoCapture('rtsp://admin:xike123456@fanmaoyang.kmdns.net:58084/stream1')
    fps = int(cap.get(5))
    print('fps:', fps)
    t = int(1000/fps)

    videoWriter = None

    while True:

        # try:
        _, frame = cap.read()
        if frame is None:
            break
        #画图  截图
        cv2.line(frame, (763, 680), (2154, 680), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(frame, (763, 680), (294, 1132), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        frame = frame[676:1380, 306:]

        t0=time.time()
        result = det.feedCap(frame)
        print('torch wasted time %5f'%(time.time()-t0))
        result_frame = result['frame']
        magic_box= result['magic_box']
        result_frame = imutils.resize(result_frame, height=500)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result1.mp4', fourcc, fps, (result_frame.shape[1], result_frame.shape[0]))


if __name__ == '__main__':
    
    main()