import cv2
import socket
import pickle
import struct
# import RRDBNet_arch as arch
import numpy as np
import time
from multiprocessing import Process, Lock
from threading import Thread, Lock
import pickle
import os
import argparse
import shutil
def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument('-lf',"--logfolder", help="Local log folder which the data should be saved to", type=str, default='test/')
    parser.add_argument('-p',"--port", help="Port reference value used", type=int, default = 62000)
    parser.add_argument('-li',"--host", help="Host which the server is running on", type=str, default='127.0.0.1')
    parser.add_argument('-mec',"--mec", help="MEC we are receiving the video stream from", type=str,  default='127.0.0.1')
    parser.add_argument('-vo',"--video_o", help="Original name of the video", type=str,  default='animals_8.mp4')
    parser.add_argument('-vs',"--video_s", help="Video received and where we are saving it to", type=str,  default='received')
    parser.add_argument('-d',"--direct", help="Direct UE to UE communication", type=bool,  default=False)



    args = parser.parse_args()

    return args



class gai_destination_client(): # Running on UE2
  
    
    def __init__(self, host = "127.0.0.1", mec='127.0.0.1',port_data = 65432, logfolder = 'test/', received_video = 'reveived_video.mp4',
                  original_video = 'original',uetoue = False):

        self.host =  host #loopback
        self.mec = mec
        self.port_data = port_data
        if uetoue == True:
            self.port_data-=1
        self.receive_frame_ts= {}
        self.logfolder = logfolder
        self.received_video = received_video
        self.scale_factor=0.20
        self.original_video = original_video
        self.uetoue = uetoue

    def run(self):
        self.server_receiver()

    def start_server(self,host,port): # TODO add thread for starting these?
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(30)  # Chờ kết nối từ người xem
        client_socket, addr = server_socket.accept()
        return client_socket,addr

    def start_client(self,host,port):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while(1):
            time.sleep(0.5)
            try:
                client_socket.connect((host, port))
                break
            except Exception as e:
                pass
        print("##########CLient connected##########")
        return client_socket

    def server_receiver(self):
        fps = 24
        upscale_factor = 4
        #size = (int(544*self.scale_factor), int(960*self.scale_factor)) # if we use GAI
        size = (int(640*self.scale_factor), int(360*self.scale_factor)) # if we use GAI
        try:
            os.makedirs(self.logfolder+"/ue2/")
        except:
            pass

        out = cv2.VideoWriter(f'{self.logfolder}/ue2/received_video.mp4',cv2.VideoWriter_fourcc(*"XVID"), fps , size)
        #out = cv2.VideoWriter('./reveived_video.mp4',cv2.VideoWriter_fourcc(*"XVID"), fps , size)

        print("connecting to MEC")
        print("connecting on:",self.mec,self.port_data)
        from_mec_socket= self.start_client(self.mec,self.port_data)
        print("client connected")

        data = b""
        payload_size = struct.calcsize("2Q")
        while True:
####################### Need to cut this out#######################
            while len(data) < payload_size:
                packet = from_mec_socket.recv(4*1024)
                if not packet:
                    break
                data += packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            try:
                if len(packed_msg_size) !=0:
                    cntrl_size = struct.unpack("2Q", packed_msg_size)[0]
                    msg_size = struct.unpack("2Q", packed_msg_size)[1]
                else:
                    msg_size = 0
            except:
                print("failed")
                print(len(packed_msg_size))
            if msg_size != 0:
                while len(data) < msg_size+cntrl_size:
                    data += from_mec_socket.recv(4*1024)
                t_start = time.time_ns()

                frame_data = data[cntrl_size:msg_size+cntrl_size]
                frame = pickle.loads(frame_data)
                if self.uetoue:
                    rec_gaistatus, cntrl_msg = pickle.loads(data[:cntrl_size])
                else:

                    cntrl_msg = pickle.loads(data[:cntrl_size])

                data = data[msg_size+cntrl_size:]
                if len(frame[0])<10 and self.uetoue:
                    print(cntrl_msg,rec_gaistatus)
                    cntrl_msg = rec_gaistatus

                
                if type(cntrl_msg)==type('str'):
                    if 'exit' in cntrl_msg:
                        print("exitting")
                        self.running = False
                        break
                    if 'terminate' in cntrl_msg:
                        if self.uetoue:
                            new_name = rec_gaistatus.split('_')[1]
                            self.save_to_file(self.logfolder+"/ue2/"+new_name)
                        else:
                            new_name = cntrl_msg.split('_')[1]
                            self.save_to_file(self.logfolder+"/ue2/"+new_name)
                        print("video finished",size_)
                        print("new name", new_name)
                        print(f'{self.logfolder}/ue2/{new_name}/received_{self.original_video}.mp4')
                        
                        self.receive_frame_ts = {}
                        out.release()
                        print("released video file")
                        shutil.move(f'{self.logfolder}/ue2/received_video.mp4',f"{self.logfolder}/ue2/{new_name}/received_video.mp4")
                        out = cv2.VideoWriter(f'{self.logfolder}/ue2/received_video.mp4',cv2.VideoWriter_fourcc(*"XVID"), fps , size)
                        continue

                
                size_ = len(frame[0]),len(frame)
                print(size_,size)

                try:
                    if (len(frame[0]),len(frame)) == (640*self.scale_factor/upscale_factor,360*self.scale_factor/upscale_factor):
                        frame = cv2.resize(frame, None, fx=4, fy=4, interpolation=cv2.INTER_AREA)
                    print("write frame",cntrl_msg,size_)
                    out.write(frame)
                    self.receive_frame_ts[cntrl_msg] = time.time_ns()
                    print("processing took:",np.round((self.receive_frame_ts[cntrl_msg]-t_start)/1000000),"ms")

                except: # close the connection
                    out.release()
                    from_mec_socket.close()
                    cv2.destroyAllWindows()
                    break
            else:
                out.release()
                from_mec_socket.close()
                #self.save_to_file(self.logfolder+"/ue2/" +new_name)
                cv2.destroyAllWindows()
                break



    def save_to_file(self,folder_name):
        try:
            os.makedirs(folder_name)
        except:
            pass
        with open(f"{folder_name}/receive_frame_ts_ue2.pickle", 'wb') as handle:
            pickle.dump(self.receive_frame_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

args = parseArguments()
inp = args.__dict__

try:
    os.makedirs(inp['logfolder'])
except:
    pass

ue2_handle = gai_destination_client( host = inp['host'],
                                     mec = inp['mec'],
                                     port_data = inp['port']+1,
                                    logfolder = inp['logfolder'],
                                    original_video=inp['video_o'],
                                    uetoue=inp['direct'])

#ue2_handle = gai_destination_client( host = "127.0.0.1",port_data = start_p+1)

ue2_handle.run()
