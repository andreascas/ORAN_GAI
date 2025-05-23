#%%
import cv2

import os
import numpy as np
fps = 20
scale_factor = 1

size = (int(640*scale_factor),int(360*scale_factor))
videos = list(os.listdir("video_source_full"))

out = cv2.VideoWriter(f'sent_videos/sent_big_0{str(scale_factor)[2:]}_{fps}.mp4',cv2.VideoWriter_fourcc(*"XVID"), fps , size)
#out = cv2.VideoWriter(f'validation3.mp4',cv2.VideoWriter_fourcc(*"XVID"), fps , size)
for video in ['big.mp4']:
    cam = cv2.VideoCapture(f"video_source_full/{video}") 
    #cam = cv2.VideoCapture(f"{video}") 
    cur_video = True
    print(f"video_source_full/{video}")
    counter = 0
    while cur_video:
        ret, frame = cam.read()
        print(counter)
        counter+=1
        if type(frame) is np.ndarray: 
            #print("orig ",len(frame[0]),len(frame),size)
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            
            out.write(frame)
            #print("post ",len(frame[0]),len(frame),size,":",size[0]*1/4,size[1]*1/4)
        else:
            print(ret)
            print(type(frame))
            cur_video = False
out.release()