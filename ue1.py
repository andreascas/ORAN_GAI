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

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument('-lf',"--logfolder", help="Local log folder which the data should be saved to", type=str, default='test/')
    parser.add_argument('-p',"--port", help="Port reference value used", type=int, default = 62000)
    parser.add_argument('-li',"--host", help="Host which the server is running on", type=str, default='127.0.0.1')
    parser.add_argument('-mec',"--mec", help="MEC we are sending the video stream to", type=str,  default='127.0.0.1')
    parser.add_argument('-video',"--video", help="Video being streamed ", type=str,  default='animals_8.mp4')
    parser.add_argument('-d',"--direct", help="Direct UE to UE communication", type=bool,  default=False)
    parser.add_argument('-g',"--ngai", help="Do not apply GAI", type=bool,  default=False)




    args = parser.parse_args()

    return args



class gai_source_client(Thread): # We are running this on UE1
    use_gai = 'False'
    running = True
    fixed_fr = 20
    def __init__(self, host = '127.0.0.1',MEC = "127.0.0.1",port_data = 65433,port_control = 65435,video ='animals_8.mp4',uetoue=False,no_apply_gai=False ):

                    #Calculate by using (No. of frames)/Video_duration in seconds  
        self.mec_ip = MEC
        self.host =  host #loopback
        self.port_data = port_data
        
        self.port_control = port_control 
        self.send_frame_ts,self.receive_cntrl_ts= {},[]
        self.not_ready = False
        self.uetoue = uetoue
        if self.uetoue:
            self.second_port_data = port_control+1
        self.no_apply_gai = no_apply_gai
        

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
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Hosting on :",self.host, self.port_data)
        server_socket.bind((self.host, self.port_data))
        server_socket.listen(30)  # Chờ kết nối từ người xem
        print("Waiting for the connections...")
        client_socket, addr = server_socket.accept()
        print("Get connection from:", addr)

        if self.uetoue:
            print("second control channel")
            print("Hosting on :",self.host, self.second_port_data)
            server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            server_socket_2.bind((self.host, self.second_port_data))
            server_socket_2.listen(30)  # Chờ kết nối từ người xem
            print("Waiting for the connections...")
            client_socket_2, addr = server_socket_2.accept()
            print("Get connection from:", addr)
        print("mec connected")

        scale_factor = 1 #0.20

        videos_folder = 'video_source_full/'
        #videos = ['sent_big_1_20.mp4']*2
        videos= ['sent_big_02_20.mp4']

        #save metadata
        ue_log = '/home/nuc/uelog.log'
        f=open(ue_log,'r')
        content = f.readlines()
        for line in content:
            if 'c-rnti=' in line:
                rnti = line.split(".")[1].split(',')[0].split('=')[1]
        hex_rnti = int(rnti,16)
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
            size = (width,height)#(width*4, height*4)
            print(width*scale_factor,height*scale_factor,":",size)
            # TODO could send width/height info here
            

            counter = 0
            while self.not_ready == False:
                time.sleep(0.1)
            print("starting now")

            while True:
                ret, frame = cam.read()
                print(ret)
                if counter == 0:
                    t_start = cam.get(cv2.CAP_PROP_POS_MSEC)
                    e_t = time.time_ns()
                    t_prev = (1/self.fixed_fr)*1000 # we fix a specified rate
                else:
                    #t_next = cam.get(cv2.CAP_PROP_POS_MSEC)
                    t_next = t_prev + (1/self.fixed_fr)*1000
                    if (t_next > t_prev):
                        #print("sleeping ",(t_next-t_prev), 'exec to far',(time.time_ns()-e_t)/1000000000, "ms")
                        time.sleep((t_next-t_prev)/1000)
                    t_prev = t_next

                frame_temp = frame


                compressed_scale_factor = scale_factor/4
                gai_status = self.use_gai
                #gai_status = "False"
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
                    if self.no_apply_gai:
                        gai_status = "False" # HACK
                    print(frame_size)

                    data,msg = pickle.dumps(frame),pickle.dumps([gai_status,counter])
                    self.send_frame_ts[counter] = time.time_ns()
                    client_socket.sendall(struct.pack("2Q", len(msg), len(data)) + msg + data)
                    counter+=1
                else:
                    placeholder_data = pickle.dumps('placeholderdata')
                    cntrl_message = pickle.dumps([f'terminate_{i+1}',-1])
                    print("terminate video")
                    client_socket.sendall(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data)
                    cam.release()
                    self.save_to_file(f'test/ue1/{i+1}')
                    self.send_frame_ts = {}
                    self.receive_cntrl_ts = []
                    if self.uetoue:
                        client_socket_2.sendall(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data)
                    time.sleep(10)
                    break
            # When all videos are run through
        placeholder_data = pickle.dumps('placeholderdata')
        cntrl_message = pickle.dumps(['exit',-1])
        print("exit")

        client_socket.sendall(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data)
        client_socket.close()
        if self.uetoue:
            client_socket_2.sendall(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data)
            client_socket_2.close()
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
                frame_data = data[:msg_size]
                data = data[msg_size:] # TODO need to extract GAI flag
                frame = frame_data #pickle.loads(frame_data)
                self.receive_cntrl_ts.append(time.time_ns())
                return frame, msg_size
            else:
                print("msg_size 0, what now?")
                return -1, -1

    def control_client(self,host,port): # TODO run as a thread
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
                if frame.decode('utf8') =='Start':
                    self.not_ready = True
                    print("received ready flag")
                    print(self.use_gai)
                if frame.decode('utf8') =='True':
                    self.use_gai = 'True'
                else:
                    self.use_gai='False'
            else:
                from_mec_socket.close()


args = parseArguments()
inp = args.__dict__
print(inp)

ue_handle = gai_source_client( host = inp['host'],
                               MEC = inp['mec'],
                               port_data = inp['port'],
                                port_control = inp['port']+2,
                                video=inp['video'],
                                uetoue=inp['direct'],
                                no_apply_gai = inp['ngai'])

ue_handle.run()
