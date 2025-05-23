# GStreamer + Python ML pipeline with GStreamer ABR streaming from UE1 to MEC (TCP version)


import threading
import cv2
import pickle
import os
import time
import struct
import socket
import subprocess
import numpy as np


def hex_to_int(hex_rnti):
    return int(hex_rnti,16)

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


#magic = b'FRM0'
UE2_PORT = 5002
FEEDBACK_PORT = 6000
MEC_PORT = 5005
# Set her eif to use TCP or UDP
CommunicationType="TCP"

UE1_IP = "192.168.100.5"
UE2_IP = "192.168.100.2"
MEC_IP = "192.168.11.30"




frame_scalar = 0.25 # compression to send video generally
comp_scalar = 0.25 # compression we improve with rescaling or GAI
frame_height = 640
frame_width = 360
fps = 24 # TODO need to consider if we should apply a slighlty higher fps or quality to push more data through....

size = (int(640*frame_scalar), int(360*frame_scalar)) 

video_formats = ['high','low']
current_format = {'profile': 'high'}

# ---------- UE1 ----------

def run_ue1_pipeline():
    import gi
    import numpy as np
    import struct

    start_send = False
    frame_interval = 1.0 / fps
    frame_counter = 0

    def feedback_listener():
        nonlocal start_send
        feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        feedback_sock.bind((UE1_IP, FEEDBACK_PORT))
        print("feedback received")
        while True:
            msg, _ = feedback_sock.recvfrom(1024)
            if msg == b'START':
                print("[UE1] Received START signal from MEC")
                start_send = True
            elif msg.startswith(b'F:'):
                try:
                    profile = msg.decode().split(':')[1]
                    if profile in video_formats:
                        current_format['profile'] = profile
                        print(f"[UE1] Updated video profile to: {profile} → {video_formats[profile]}")
                except Exception as e:
                    print(f"[UE1] Failed to parse profile: {e}")

    threading.Thread(target=feedback_listener, daemon=True).start()

    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib

    Gst.init(None)

    pipeline = Gst.parse_launch("""
        filesrc location=sent_videos/sent_big_1_20.mp4 ! decodebin ! videorate ! video/x-raw,framerate=30/1 ! \
        videoconvert ! video/x-raw,format=BGR ! appsink name=mysink emit-signals=true sync=false max-buffers=1 drop=true
    """)

    appsink = pipeline.get_by_name("mysink")
    appsink.set_property("emit-signals", True)

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((MEC_IP, MEC_PORT))

    log_data = {'ts':[],'frame_id':[]}

    def save_to_file():
        os.makedirs('abr_ue1_data', exist_ok=True)
        with open("abr_ue1_data/log_data_ue1.pickle", 'wb') as handle:
            pickle.dump(log_data, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    def on_new_sample(sink):
        nonlocal  frame_counter,log_data, start_send
        if not start_send:
            print("fuck?")
            return Gst.FlowReturn.OK
        start_time = time.time()
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame_data = map_info.data
        buffer.unmap(map_info)
        frame = np.frombuffer(frame_data, np.uint8).reshape((int(frame_width),
                                                              int(frame_height), 3))

        frame = cv2.resize(frame, ( int(frame_height*frame_scalar),
                                    int(frame_width*frame_scalar)),
                                    interpolation=cv2.INTER_AREA)
        frame_ts = time.time()*1e9
        frame_id = frame_counter
        frame_counter += 1

        profile = current_format['profile']
        if profile == 'low':
            width, height = int(frame_width*frame_scalar*comp_scalar), int(frame_height*frame_scalar*comp_scalar)
            frame = cv2.resize(frame, ( int(frame_height*frame_scalar*comp_scalar),
                                        int(frame_width*frame_scalar*comp_scalar)),
                                        interpolation=cv2.INTER_AREA)
        else:
            width, height = int(frame_width*frame_scalar), int(frame_height*frame_scalar)

        encoded = pickle.dumps(frame)
        packet = struct.pack('!HHI', width, height, frame_id) + encoded
        packet_len = struct.pack('!I', len(packet))

        try:
            if CommunicationType =='TCP':
                sock.sendall(packet_len + packet)
                log_data['ts'].append(frame_ts)
                log_data['frame_id'].append(frame_id)
            elif CommunicationType =='UDP' and len(packet) < 65507:
                sock.sendto(packet,(MEC_IP, MEC_PORT))
                log_data['ts'].append(frame_ts)
                log_data['frame_id'].append(frame_id)
            else:
                print("packet too big for udp:",len(packet))
        except Exception as e:
            print(e)
            print("Failed to send packet")

        elapsed = time.time() - start_time
        time.sleep(max(0, frame_interval - elapsed))
        return Gst.FlowReturn.OK

    def on_message(bus, message):
        if message.type == Gst.MessageType.EOS:
            print("End of video stream.")
            loop.quit()

    while not start_send:
        print("waiting to start")
        time.sleep(1)

    appsink.connect("new-sample", on_new_sample)
    pipeline.set_state(Gst.State.PLAYING)
    bus.connect("message", on_message)
    print("UE1 pipeline running...")
    started = time.time()
    loop = GLib.MainLoop()
    loop.run()
    sock.close()
    ended = time.time()
    print("ran for:",ended-started,"seconds")
    save_to_file()


# ---------- MEC ----------
def run_mec_pipeline():
    from monitor import monitor
    def save_to_file():
        os.makedirs('abr_mec_data', exist_ok=True)
        with open("abr_mec_data/log_data_mec.pickle", 'wb') as f:
            pickle.dump(log_data, f)

    def collect_metrics(ip1,ip2,location1,location2,logdir):
        # ue 1 on ip1
        try:
            os.makedirs(logdir)
        except:
            pass
        command_copy = f"scp -r -i /home/computing/.ssh/sharekey.ed25519 nuc@{ip1}:/home/nuc/ORAN_GAI_VIDEO_enhance/{location1}/ue1/ {logdir}"
        subprocess.run(command_copy, shell = True, executable="/bin/bash")

        command_copy = f"scp -r -i /home/computing/.ssh/sharekey.ed25519 nuc@{ip2}:/home/nuc/ORAN_GAI_VIDEO_enhance/{location2}/ue2/ {logdir}"
        subprocess.run(command_copy, shell = True, executable="/bin/bash")


    import threading
    if UE1_IP != "127.0.0.1":
        # this is compiled on the computer with the flexric, so must be compiled and copied over
        from util import xapp_sdk as ric

        ric.init()

    log_data = {'ts_recv':[],'frame_id':[],'ts_send':[]}

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((MEC_IP, MEC_PORT))
    server_sock.listen(1)
    feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def use_gai(rnti,monitor):

        high_snr = 30
        
        if UE1_IP != '127.0.0.1':
            pusch = monitor.get_pusch(rnti)
            print("startup",pusch)
        if pusch < high_snr:
            print("low snr",pusch,high_snr)
            return "True"
        else:
            print("high snr",pusch,high_snr)
            return "False"



    def start_signal(monitor_):
        rnti = get_rnti(UE1_IP)
        prev_flag = True
        while True:
            use_gai_flag = use_gai(rnti,monitor_)

            if  use_gai_flag == "True" and prev_flag == "False":  # we start traffic when everything is connected, on first low channel
                break
            else:
                prev_flag = use_gai_flag
        feedback_sock.sendto(b'START', (UE1_IP, FEEDBACK_PORT))
        print("[MEC] Sent START signal to UE1")
        return


    conn, _ = server_sock.accept()
    print("connected to ue 1")

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect((UE2_IP, UE2_PORT))
    print("connected to ue2")
    monitor_ = monitor('abr_mec_data_ran',False)
    threading.Thread(target=start_signal,args=[monitor_], daemon=True).start()

    try:
        while True:
            length_data = conn.recv(4)
            if not length_data:
                break
            packet_len = struct.unpack('!I', length_data)[0]
            packet = b''
            while len(packet) < packet_len:
                chunk = conn.recv(packet_len - len(packet))
                if not chunk:
                    break
                packet += chunk

            if len(packet) < 8:
                continue
            width, height, frame_id = struct.unpack('!HHI', packet[0:8])
            print(frame_id,len(packet))
            encoded_frame = packet[8:]

            log_data['ts_recv'].append(time.time()*1e9)
            log_data['frame_id'].append(frame_id)

            header = struct.pack('!HHI', width, height, frame_id)
            fwd_packet = header + encoded_frame
            fwd_len = struct.pack('!I', len(fwd_packet))
            try:
                client_sock.sendall(fwd_len + fwd_packet)
                log_data['ts_send'].append(time.time()*1e9)
            except:
                break
    finally:
        conn.close()
        client_sock.close()
        server_sock.close()
        save_to_file()
        print("tring to collect metrics")
        try:
            collect_metrics(UE1_IP,UE2_IP,'abr_ue1_data/','abr_ue2_data/',"abr_mec_data/")
        except:
            pass



# ---------- UE2 ----------
def run_ue2_receiver():
    import threading
    logdata = {'ts':[],'frame_id':[]}
    import numpy as np
    bitrate_lock = threading.Lock()
    feedback_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    current_profile = {'value': 'low'}
    out = cv2.VideoWriter(f'abr_ue2_data/received_video.mp4',cv2.VideoWriter_fourcc(*"XVID"), fps , size)

    from collections import deque

    bitrate_window = deque()
    bitrate_window_duration = 5  # seconds

    bitrate_window = deque()
    bitrate_window_duration = 5  # seconds
    def abr_feedback_loop():
        is_probing = False
        last_probe_time = 0
        probe_duration = 0.5
        probe_interval = 5
        threshold_kbps = 6000

        while True:
            now = time.time()
            if is_probing:
                time.sleep(probe_duration)  # Actively wait during probe period
                now = time.time()
                with bitrate_lock:
                    probe_bits = sum(size for ts, size in bitrate_window if now - ts <= probe_duration)
                probe_kbps = (probe_bits * 8) / probe_duration / 1000
                print(f"[UE2] Probe bitrate: {probe_kbps:.1f} kbps")

                if probe_kbps < threshold_kbps:
                    print("[UE2] Probe failed → revert to low")
                    feedback_sock.sendto(b'F:low', (UE1_IP, FEEDBACK_PORT))
                    current_profile['value'] = 'low'
                else:
                    print("[UE2] Probe succeeded → stay high")
                    current_profile['value'] = 'high'

                is_probing = False
                last_probe_time = now
                time.sleep(0.1)  # Allow recovery time before measuring again
            else:
                with bitrate_lock:
                    now = time.time()
                    while bitrate_window and now - bitrate_window[0][0] > bitrate_window_duration:
                        bitrate_window.popleft()
                    bits = sum(size for _, size in bitrate_window)

                window_time = bitrate_window[-1][0] - bitrate_window[0][0] if len(bitrate_window) > 1 else 1
                bitrate_kbps = (bits * 8) / window_time / 1000
                print(f"[UE2] Measured bitrate: {bitrate_kbps:.1f} kbps")

                # ▼▼▼ DOWNGRADE if bitrate drops while in high ▼▼▼
                if current_profile['value'] == 'high' and bitrate_kbps < threshold_kbps:
                    print("[UE2] Bitrate dropped → downgrade to low")
                    feedback_sock.sendto(b'F:low', (UE1_IP, FEEDBACK_PORT))
                    current_profile['value'] = 'low'

                # ▲▲▲ EXISTING UPGRADE PROBE LOGIC ▲▲▲
                if current_profile['value'] == 'low' and (now - last_probe_time > probe_interval):
                    print("[UE2] Starting probe")
                    feedback_sock.sendto(b'F:high', (UE1_IP, FEEDBACK_PORT))
                    is_probing = True

            time.sleep(0.1)

    threading.Thread(target=abr_feedback_loop, daemon=True).start()
    if CommunicationType=='TCP':
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((UE2_IP, UE2_PORT))
    server_sock.listen(1)
    conn, _ = server_sock.accept()

    def save_to_file():
        os.makedirs('abr_ue2_data', exist_ok=True)
        with open("abr_ue2_data/logdata_ue2.pickle", 'wb') as f:
            pickle.dump(logdata, f)

    while True:
        length_data = conn.recv(4)
        if not length_data:
            continue
        try:
            packet_len = struct.unpack('!I', length_data)[0]
        except:
            print(length_data)
            continue
        packet = b''
        while len(packet) < packet_len:
            chunk = conn.recv(packet_len - len(packet))
            if not chunk:
                break
            packet += chunk

        if len(packet) < 12:
            continue
        frame_ts = time.time()*1e9
        width, height, frame_id = struct.unpack('!HHI', packet[0:8])
        encoded_frame = packet[8:]

        frame = pickle.loads(encoded_frame)
        if (width, height) != (int(frame_height*frame_scalar), int(frame_width*frame_scalar)):
            frame = cv2.resize(frame, (int(frame_height*frame_scalar), int(frame_width*frame_scalar)), interpolation=cv2.INTER_LINEAR)


        if frame is not None:
            with bitrate_lock:
                bitrate_window.append((time.time(), len(packet)))
            logdata['ts'].append(frame_ts)
            logdata['frame_id'].append(frame_id)
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("break at cv2 ")
            break
    conn.close()
    server_sock.close()
    save_to_file()
    out.release()