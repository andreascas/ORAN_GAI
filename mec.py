import cv2
import socket
import pickle
import struct
# import RRDBNet_arch as arch
import torch
import numpy as np
import time
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from threading import Thread, Lock
from util import xapp_sdk as ric
import os
import subprocess
import RRDBNet_arch as arch

import argparse

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument('-lf',"--logfolder", help="Local log folder which the data should be saved to", type=str, default='test/')
    parser.add_argument('-p',"--port", help="Port reference value used", type=int, default = 62000)
    parser.add_argument('-li',"--host", help="Host which the server is running on", type=str, default='127.0.0.1')
    parser.add_argument('-ue1',"--user1", help="UE 1 host who we receive video from", type=str,  default='127.0.0.1')
    parser.add_argument('-ue2',"--user2", help="UE 2 host who we forward video to", type=str,  default='127.0.0.1')
    parser.add_argument('-rnti',"--rnti", help="RNTI of user 1", type=str,  default='0xffff')
    parser.add_argument('-d',"--direct", help="Direct UE to UE communication", type=bool,  default=False)

    args = parser.parse_args()

    return args


device = 'cuda'


## fast model
# model_path = 'realesr-general-x4v3.pth'  
# model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'))['params'], strict=True)

def hex_to_int(hex_rnti):
    return int(hex_rnti,16)

def int_to_hex(int_rnti):
    return hex(int_rnti)


class gai_server(): # We are running this on MEC
    # Get the Default resolutions
    running = True
    fps = 20       #Put the fps value as your convenience or 
    #scale_factor = 0.125
    scale_factor = 0.2
    #scale_factor = 0.15
    verbose = False
    size = (int(640*scale_factor), int(360*scale_factor)) # if we use GAI
    startup_period = True
    
    def __init__(self, host = '127.0.0.1', ue1 = "127.0.0.1",port_data1 = 65432,
                 ue2 = "127.0.0.1", port_data2 = 65433,
                  port_control = 65434,rnti=None,logfolder = 'test/',uetoue=False):
        
        
        if ue1 != '127.0.0.1':
            ric.init()
            self.conn = ric.conn_e2_nodes()
            self.monitor = monitor(logfolder,uetoue)
        

        self.logfolder = logfolder

        self.node_idx = 0
        self.host = host

        
        self.port_data1 = port_data1
        self.port_data2 = port_data2
        if uetoue:
            self.port_rec2 = port_control+1  

        self.host1 =  ue1 #loopback
        self.port_control = port_control 
        self.host2 = ue2

        self.mutex = Lock()
        self.forward_frame_ts,self.receive_frame_ts= {i:-1 for i in range(12500)} ,{i:-1 for i in range(12500)}
        self.cur_gai = False
        self.rnti=rnti
        self.ready = False
        self.uetoue = uetoue


    def run(self):

        self.receiver = Thread(target=self.server_receiver)
        self.receiver.start()
        print("started receiver")
        self.control_established = False
        self.gai_checker = Thread(target=self.control_server)
        self.gai_checker.start()
        print("Gai checker started, for rnti",self.rnti)
        while self.running:
            if self.control_established==True:
                break


    def start_server(self,host,port): # TODO add thread for starting these?
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(30)  # Chờ kết nối từ người xem
        client_socket, addr = server_socket.accept()
        return client_socket,addr

    def start_client(self,host,port):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        return client_socket


    def apply_gai(self,frame):
        try:
            img = frame * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)
            with torch.no_grad():
                output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))

            output = np.ascontiguousarray(output)
            frame = (output * 255).astype(np.uint8)
            return frame,output
        except Exception as e:
            print("error ", e)
            #cv2.destroyAllWindows()
            return -1,-1
        
        


    def control_server(self):
        # We integrate the xApp feature in here
        print("starting server \nwaiting for connection...")
        to_ue1_control_socket, addr = self.start_server(self.host,self.port_control)
        print("connected to", addr,'control server')
        self.control_established = True
        while self.running:
            time.sleep(0.1) # every 100 ms
            use_gai_flag, value = self.use_gai(self.rnti)
            tmp_status = bytes(use_gai_flag,encoding='utf8') #self.use_gai()

            if self.cur_gai != use_gai_flag:                    

                if self.ready == False:
                    to_ue1_control_socket.sendall(struct.pack("Q", len(tmp_status)) + tmp_status)
                self.cur_gai = use_gai_flag
                if self.ready == True and use_gai_flag == "True":  # we start traffic when everything is connected, on first low low channel
                    self.ready = False
                    msg = bytes('Start',encoding='utf8')
                    to_ue1_control_socket.sendall(struct.pack("Q", len(msg)) + msg)
                    msg = bytes('True',encoding='utf8')
                    to_ue1_control_socket.sendall(struct.pack("Q", len(msg)) + msg)
                

        to_ue1_control_socket.close()
        if self.host1 != '127.0.0.1':
            self.monitor.shutdown()

        
    def use_gai(self,rnti):
        high_snr = 28
        high_snr = 26
        high_snr = 28 # with fps 20 at 0.1 reduction
        bleh = time.time() % 20
        bleh = 20
        pusch = bleh
        
        if self.uetoue:
            high_snr = 27
        if self.ready:
            high_snr = 35
            
        if self.host1 != '127.0.0.1':
            pusch = self.monitor.get_pusch(rnti)
        if self.startup_period:
            print(pusch)
        if pusch < high_snr:
            #print("Apply GAI")
            print(pusch)
            return "True", pusch
        else:
            print(pusch)
            #print("Do not apply GAI")
            return "False", pusch

        
    def server_receiver(self): # placeholder to simply test if UE-> MEC works
        #model_path = 'realesr-general-x4v3.pth'  
        #self.model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        #self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'))['params'], strict=True)
        self.model_path = 'models/RRDB_PSNR_x4.pth'
        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval()
        self.device = 'cuda'
        self.model = self.model.to(device)
        print("connecting to UE2")
        if self.uetoue ==False:
            to_ue2_socket,ue2addr= self.start_server(self.host,self.port_data2)
            client_socket =self.start_client(self.host1, self.port_data1)
        else:
            client_socket =self.start_client(self.host1, self.port_rec2)
             
        print("connecting to UE1")
        
        data = b""
        payload_size = struct.calcsize("2Q")
        size_ =0,0
        self.ready = True
        if self.verbose:
            print("going into self.running")
        if self.uetoue == True:
            print("no forwarding done")
        while self.running:
            received_correctly = True
####################### Need to cut this out#######################
            while len(data) < payload_size:
                packet = client_socket.recv(4*1024)
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
            if msg_size != 0:
                self.startup_period = False
                while len(data) < msg_size+cntrl_size:
                    data += client_socket.recv(4*1024)
                if self.verbose:
                    print("new msg received")
                frame_data = data[cntrl_size:msg_size+cntrl_size]

                cntrl_msg_rec = pickle.loads(data[:cntrl_size])
                rec_gaistatus, packet_counter = cntrl_msg_rec
                data = data[msg_size+cntrl_size:]
                if 'terminate' in rec_gaistatus:
                    print("video finished",size_)
                    new_name = rec_gaistatus.split('_')[1]
                    self.save_to_file(self.logfolder+"/mec/"+new_name)
                    self.forward_frame_ts,self.receive_frame_ts= {i:-1 for i in range(12500)} ,{i:-1 for i in range(12500)}
                    placeholder_data = pickle.dumps('placeholderdata')
                    cntrl_message = pickle.dumps(f'terminate_{new_name}')
                    if self.uetoue == False:
                        to_ue2_socket.sendall(struct.pack("2Q", len(cntrl_message), len(placeholder_data)) + cntrl_message + placeholder_data)
                    continue # we continue into the new packet we receive
                if rec_gaistatus == 'exit':
                    print("exitting")
                    self.running = False
                    break

                try:
                    frame = pickle.loads(frame_data)
                except:
                    received_correctly = False
                    print("failed to pickle")
                    pass
                if received_correctly:
                    rx_t = time.time_ns()
                    self.receive_frame_ts[packet_counter] = time.time_ns()
####################### Need to cut this out#######################
                    apply_gai = rec_gaistatus 
                    if apply_gai == 'True': 
                        try:
                            if self.verbose:
                                print("applying GAI")
                            pre_gai = time.time_ns()
                            frame,output = self.apply_gai(frame)
                            post_gai = time.time_ns()
                        except Exception as e:
                            print('error applying GAI\n', e)
                            break
                    else:
                        post_gai = 0
                    frame = pickle.dumps(frame)
                    cntrl_msg = pickle.dumps(packet_counter)
                    if self.verbose:
                        print("Forwarding to ue2")
                    
                    to_ue2_socket.sendall(struct.pack("2Q", len(cntrl_msg),len(frame)) + cntrl_msg + frame)
                    tx_t = time.time_ns()

                    self.forward_frame_ts[packet_counter]= tx_t

                    print("postgai:",np.round((post_gai-rx_t)/1000000,2),", pretrans:",np.round((tx_t-rx_t)/1000000,2))

        client_socket.close()
        if self.uetoue == False:
            to_ue2_socket.close()
        self.running = False

        if self.host1 != '127.0.0.1':
            self.collect_metrics(self.host1,self.host2,'test/','test/',self.logfolder)

    def collect_metrics(self,ip1,ip2,location1,location2,logdir):
        # ue 1 on ip1
        try:
            os.makedirs(logdir)
        except:
            pass
        command_copy = f"scp -r -i /home/computing/.ssh/sharekey.ed25519 nuc@{ip1}:/home/nuc/ORAN_GAI_VIDEO_enhance/{location1}/ue1/ {logdir}"
        subprocess.run(command_copy, shell = True, executable="/bin/bash")

        command_copy = f"scp -r -i /home/computing/.ssh/sharekey.ed25519 nuc@{ip2}:/home/nuc/ORAN_GAI_VIDEO_enhance/{location2}/ue2/ {logdir}"
        subprocess.run(command_copy, shell = True, executable="/bin/bash")


    
    def save_to_file(self,folder_name):
        try:
            os.makedirs(folder_name)
        except:
            pass
        with open(f"{folder_name}/forward_frame_ts_mec.pickle", 'wb') as handle:
            pickle.dump(self.forward_frame_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{folder_name}/receive_frame_ts_mec.pickle", 'wb') as handle:
            pickle.dump(self.receive_frame_ts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open(f"{folder_name}/send_cntrl_mec.pickle", 'wb') as handle:
        #     pickle.dump(self.send_cntrl, handle, protocol=pickle.HIGHEST_PROTOCOL)


# xApp monitoring
from sm_api import monitor_app

class monitor():

    def __init__(self,logfolder,uetoue=False):
        if uetoue:
            self.average_count = 500
        else:
            self.average_count = 500
        self.logfolder=logfolder
        self.monitor_handle = monitor_app.monitor() 
        self.mac_cb = self.monitor_handle.mac_cb

    def get_pusch(self,rnti):
        pusch_snr =-1
        if rnti in self.mac_cb.test:
            d = self.mac_cb.test[rnti]['pusch_snr'].copy()
            d = list(d)
            #d.rotate(self.average_count)
            pusch_snr = d[-self.average_count:]
            mean_snr = np.mean(pusch_snr)
        return mean_snr
    def get_pucch(self,rnti):
        pucch_snr = -1
        if rnti in self.mac_cb.test:
            pucch_snr = self.mac_cb.test[rnti]['pucch_snr'][-1]
        return pucch_snr
    def get_ul_mcs(self,rnti):
        ul_mcs1 = -1
        if rnti in self.mac_cb.test:
            ul_mcs1 = self.mac_cb.test[rnti]['ul_mcs1'][-1]
        return ul_mcs1
    
    def shutdown(self):

        self.monitor_handle.shutdown(self.logfolder)
        while ric.try_stop == 0:
            time.sleep(1)

def get_rnti(ip):
    command_copy = f"scp -i /home/computing/.ssh/sharekey.ed25519 nuc@{ip}:/home/nuc/uelog.log '/home/computing/Andreas/ORAN_GAI_VIDEO_enhance'"
    subprocess.run(command_copy, shell = True, executable="/bin/bash")

    # copy uelog file
    ue_log = '/home/computing/Andreas/ORAN_GAI_VIDEO_enhance/uelog.log'
    f=open(ue_log,'r')

    content = f.readlines()

    for line in content:
        if 'c-rnti=' in line:
            rnti = line.split(".")[1].split(',')[0].split('=')[1]

    return str(hex_to_int(rnti))

args = parseArguments()
inp = args.__dict__

if inp['user1'] != '127.0.0.1':
    rnti = get_rnti(inp['user1'])
else:
    rnti = inp['rnti']


route_ue1 = f"echo {'computingpc'} | sudo ip route add {inp['user1']} via 192.168.10.15 dev eno1 onlink"
route_ue2 = f"echo {'computingpc'} | sudo ip route add {inp['user2']} via 192.168.10.15 dev eno1 onlink"


mec_handle = gai_server(host = inp['host'],ue1 = inp['user1'],port_data1 = inp['port'],
                 ue2 = inp['user2'], port_data2 = inp['port']+1,
                  port_control = inp['port']+2, logfolder=inp['logfolder'],rnti=rnti,
                  uetoue=inp['direct'])

mec_handle.run()