# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 13:32:10 2021

@author: sunjh
"""

import numpy as np
fx = np.poly1d(np.load(r"weights/fx.npy"))
fy = np.poly1d(np.load(r"weights/fy.npy"))
fz = np.poly1d(np.load(r"weights/fz.npy"))
# def output_lane(x,y):
#
#
#
#       d = {'6': '深水区第一泳道', '5': "深水区第二泳道", '4': "深水区第三泳道", '3': "深水区第四泳道",
#                 '2': "深水区第五泳道", '1': "深水区第六泳道", '16': '浅水区第一泳道', '15': '浅水区第二泳道',
#                 '14': '浅水区第三泳道', '13': '浅水区第四泳道', '12': '浅水区第五泳道', '11': '浅水区第六泳道', '0': '各泳道一切正常！'}
#
#       f1 =  np.poly1d(np.load(r"/deployment/chongming_deployment0714/apps/camera/model_data/f1.npy"))
#       f2 =  np.poly1d(np.load(r"/deployment/chongming_deployment0714/apps/camera/model_data/f2.npy"))
#       f6 =  np.poly1d(np.load(r"/deployment/chongming_deployment0714/apps/camera/model_data/f6.npy"))
#       f7 =  np.poly1d(np.load(r"/deployment/chongming_deployment0714/apps/camera/model_data/f7.npy"))
#       f8 =  np.poly1d(np.load(r"/deployment/chongming_deployment0714/apps/camera/model_data/f8.npy"))
#       f9 =  np.poly1d(np.load(r"/deployment/chongming_deployment0714/apps/camera/model_data/f9.npy"))
#       f10 =  np.poly1d(np.load(r"/deployment/chongming_deployment0714/apps/camera/model_data/f10.npy"))
#
#       if x > 1600:
#             if y > f1(x):
#                   lane = '1'
#             else:
#                  if  y>714:
#                        lane = '2'
#                  else:
#                      if y>626:
#                            lane = '3'
#                      else:
#                           if  y>f7(x):
#                              lane = '4'
#                           else:
#                              if y>f9(x):
#                                    lane = '5'
#                              else:
#                                    lane = '6'
#
#       else:
#             if y > f2(x):
#                   lane = '11'
#             else:
#                  if y>736:
#                        lane = '12'
#                  else:
#                        if y >f6(x):
#                              lane= '13'
#                        else:
#                              if y>f8(x):
#                                    lane= '14'
#                              else:
#                                    if y>f10(x):
#                                          lane = '15'
#                                    else:
#                                          lane = '16'
#
#
#       return lane


def distance(x, y,width,height):
    import numpy as np
    x= x*width
    y= y*height
    if np.abs(fx(x)-y)<85:
        return -1
    else:
        pred_y = fy(x)
        # print(pred_y, y, x)
        if  np.abs(pred_y-y)<89:
            # print(pred_y, y, x)
            return -1
        else:
             return 0

def beyond(x,y):
    # x = x * 2560
    # y = y * 1440
    pred_y = fz(x)
    if y<pred_y:
        # print(pred_y, y, x)
        return 0
    else:
        return 1



if __name__ == "__main__":
    #2254 704
      print(distance(0.8331854480922803,0.03370351239669423,2254,704))
      print(beyond(268,220))
