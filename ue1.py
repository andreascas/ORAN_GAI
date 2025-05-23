import cv2
import socket
import pickle
import struct
import os
import numpy as np
from multiprocessing import Process, Lock
from threading import Thread,Lock
import time
import pickle

import argparse
CommunicationType="UDP"

def parseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-lf',"--logfolder", help="Local log folder which the data should be saved to", type=str, default='test/')
    parser.add_argument('-p',"--port", help="Port reference value used", type=int, default = 62000)
    parser.add_argument('-li',"--host", help="Host which the server is running on", type=str, default='127.0.0.1')
    parser.add_argument('-mec',"--mec", help="MEC we are sending the video stream to", type=str,  default='127.0.0.1')
    parser.add_argument('-video',"--video", help="Video being streamed ", type=str,  default='animals_8.mp4')
    parser.add_argument('-nogai',"--nogai", help="Disable GAI, MEC forwards the frames without upscaling", type=bool,  default=False)
    parser.add_argument('-protocol',"--CommunicationType", help="Setting if to use TCP or UDP", type=str,  default='UDP')


    args = parser.parse_args()

    return args



class gai_source_client(Thread): # We are running this on UE1
    use_gai = 'False'
    running = True
    fixed_fr = 24 # framerate, time between 
    def __init__(self, host = '127.0.0.1',MEC = "127.0.0.1",port_data = 65433,port_control = 65435,video ='animals_8.mp4',nogai=False ):

        verbose_states = ['offline','silent','simple','debug']
        self.verbose_state = 'silent'
        if self.verbose_state not in verbose_states:
            print("wrong verbose state not implemented")
        self.mec_ip = MEC
        self.host =  host 
        self.port_data = port_data
        
        self.port_control = port_control 
        self.send_frame_ts,self.receive_cntrl_ts= {},[]
        self.not_ready = False

        self.nogai = nogai
        

    def run(self):
        self.gai_control_client = Thread(target=self.control_client,args=(self.mec_ip,self.port_control))
        self.gai_control_client.start()
        print("started control client")

        self.video_sender = Thread(target=self.data_server)
        self.video_sender.start()
        print("started video sender")


    def wait_until(self,ts):
        now=time.time_ns()/1000000
        print("waiting",ts-now,"ms")
        if ts > now:
            time.sleep((ts-now)/1000)
            return
        else:
            return

    def data_server(self): #TODO make it a thread

        cntrl_messages = ['terminate','exit','setup']
        if CommunicationType=="TCP":
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((self.host, self.port_data))
            print("TCP stuff")
        else:
            print("UDP STUFF")
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
        print("Hosting on :",self.host, self.port_data,"with",CommunicationType)
        
        
        if CommunicationType=="TCP":
            print("TCP stuff")
            server_socket.listen(30)  # Chờ kết nối từ người xem
            print("Waiting for the connections...")
            client_socket, addr = server_socket.accept()
            print("Get connection from:", addr)
        else:
            client_socket = server_socket

        print("mec connected")

        scale_factor = 0.25 # For running it in our setup we downscaled these generally to pass through the RAN
        videos= [video]

        #save metadata
        ue_log = '/home/nuc/uelog.log'
        f=open(ue_log,'r')
        content = f.readlines()
        for line in content:
            if 'c-rnti=' in line:
                rnti = line.split(".")[1].split(',')[0].split('=')[1]
        try: 
            hex_rnti = int(rnti,16)
        except:
            hex_rnti = 0
        meta_data = {'videos':videos,'rnti':hex_rnti,
                     'scale_factor':scale_factor}
        try:
            os.makedirs('test/ue1/')
        except:
            pass
        
        with open(f"{'test/ue1/'}/metadata.pickle", 'wb') as handle:
            pickle.dump(meta_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

        for i in range(len(videos)):
            video = "sent_videos/"+videos[i]

            print(video)
            cam = cv2.VideoCapture(video) 
            width = int(cam.get(3))
            height = int(cam.get(4))
            size = (width,height)
            print(width*scale_factor,height*scale_factor,":",size)
            
            counter = 0
            if self.verbose_state == 'offline':
                self.not_ready = True
            while self.not_ready == False:
                time.sleep(0.1)
            print("starting now")

            while True:
                ret, frame = cam.read()
                if counter == 0:
                    t_prev = (1/self.fixed_fr)*1000 # we fix a specified rate
                else:
                    t_next = t_prev + (1/self.fixed_fr)*1000
                    if (t_next > t_prev):
                        time.sleep((t_next-t_prev)/1000)
                    t_prev = t_next

                frame_temp = frame


                compressed_scale_factor = scale_factor/4
                gai_status = self.use_gai
                if type(frame_temp) is np.ndarray: 
                    if gai_status =="False":
                        frame = cv2.resize(frame_temp, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                        frame_size = (width*scale_factor,height*scale_factor)
                    elif gai_status=="True":
                        frame = cv2.resize(frame_temp, None, fx=compressed_scale_factor, fy=compressed_scale_factor, interpolation=cv2.INTER_AREA)
                        frame_size = (width*compressed_scale_factor,height*compressed_scale_factor)
                    else:
                        print("status flag is wrong",gai_status)
                        break
                    if self.nogai:
                        gai_status = "False" # HACK
                    data,msg = pickle.dumps(frame),pickle.dumps([gai_status,counter])
                    self.send_frame_ts[counter] = time.time_ns()
                    if CommunicationType == "TCP":
                        client_socket.sendall(struct.pack("2Q", len(msg), len(data)) + msg + data)
                    else:
                        print("Sending data to ",(self.mec_ip, self.port_data))
                        client_socket.sendto((struct.pack("2Q", len(msg), len(data)) + msg + data),(self.mec_ip, self.port_data))
                    counter+=1
                else:
                    placeholder_data = pickle.dumps('placeholderdata')
                    cntrl_message = pickle.dumps([f'terminate_{i+1}',-1])
                    print("terminate video")
                    if CommunicationType == "TCP":
                        client_socket.sendall(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data)
                    else:
                        client_socket.sendto(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data,(self.mec_ip, self.port_data))
                        print("Sending data to ",(self.mec_ip, self.port_data))
                    
                    cam.release()
                    self.save_to_file(f'test/ue1/{i+1}')
                    self.send_frame_ts = {}
                    self.receive_cntrl_ts = []
                    time.sleep(10)
                    break
            # When all videos are run through
        placeholder_data = pickle.dumps('placeholderdata')
        cntrl_message = pickle.dumps(['exit',-1])
        print("exit")
        if CommunicationType=="TCP":
            client_socket.sendall(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data)
        else:
            client_socket.sendto(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data,(self.mec_ip,self.port_data))
        client_socket.close()

        print("running exitted")
        self.running = False


    def save_to_file(self,folder_name):
        try:
            os.makedirs(folder_name)
        except:
            pass

        with open(f"{folder_name}/send_frame_ts_ue1.pickle", 'wb') as handle:
            pickle.dump(self.send_frame_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{folder_name}/receive_cntrl_ts_ue1.pickle", 'wb') as handle:
            pickle.dump(self.receive_cntrl_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

    def gen_receiver(self,from_mec_socket):
        payload_size = struct.calcsize("Q")
        data = b""
        while True:
            while len(data) < payload_size:
                packet = from_mec_socket.recv(4*1024)
                if not packet:
                    break
                data += packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            try:
                if len(packed_msg_size) !=0:
                    msg_size = struct.unpack("Q", packed_msg_size)[0]
                else:
                    msg_size = 0
            except:
                pass
            if msg_size != 0:
                while len(data) < msg_size:
                    data += from_mec_socket.recv(4*1024)
                print("got a control message!")
                frame_data = data[:msg_size]
                data = data[msg_size:] 
                frame = frame_data 
                self.receive_cntrl_ts.append(time.time_ns())
                return frame, msg_size
            else:
                print("msg_size 0, what now?")
                return -1, -1

    def control_client(self,host,port): 
        from_mec_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"control client connecting on {host},{port}")
        while True:
            try:
                from_mec_socket.connect((host, port))
                print("connected with",host,'control client')
                break
            except:
                pass
        print("connected to the control server")

        while self.running:
            frame, msg_size = self.gen_receiver(from_mec_socket)
            if frame != -1:
                print(frame.decode('utf8'),'here stupid')
                print("controller:",frame.decode('utf8'))
                if frame.decode('utf8') =='Start':
                    self.not_ready = True
                    print("received ready flag")
                    print(self.use_gai)
                if frame.decode('utf8') =='True':
                    self.use_gai = 'True'
                else:
                    self.use_gai='False'
            else:
                print("####################closed feedback###############")
                from_mec_socket.close()


args = parseArguments()
inp = args.__dict__
print(inp)

CommunicationType = inp['CommunicationType']

ue_handle = gai_source_client( host = inp['host'],
                               MEC = inp['mec'],
                               port_data = inp['port'],
                                port_control = inp['port']+2,
                                video=inp['video'],
                                nogai = inp['nogai'])

ue_handle.run()
