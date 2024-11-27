
# %%import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import tikzplotlib as tikz


#  similarity scoring
####################### Image processing tools ####################### 

def calculate_psnr(frame1, frame2):
    # mse = np.mean((frame1 - frame2) ** 2)
    # if mse == 0:
    #     return float('inf')
    max_pixel = 255.0
    # psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    mse_bands = []
    for i in range(frame1.shape[2]):
        mse_bands.append(np.mean(np.square(frame1[:, :, i] - frame2[:, :, i])))

    psnr =  20 * np.log10(max_pixel) - 10.0 * np.log10(np.mean(mse_bands))
    return psnr

def calculate_average_psnr(video_path1, video_path2):
    video1 = cv2.VideoCapture(video_path1)
    video2 = cv2.VideoCapture(video_path2)
    psnr_values = []

    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if not ret1:
            print("original video not working",ret1)
        if not ret2:
            print("new video not working", ret2)

        # Break the loop if either video ends
        if not ret1 or not ret2:
            break
        # Resize frame2 to match the shape of frame1
        frame2_resized = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        # Calculate PSNR for the frame pair
        psnr = calculate_psnr(frame1, frame2_resized)
        psnr_values.append(psnr)

    video1.release()
    video2.release()

    return np.mean(psnr_values), np.array(psnr_values)

def plot_psnr_cdf(psnr_values, label, plot=False):
    sorted_psnr = np.sort(psnr_values)
    cdf = np.arange(1, len(sorted_psnr) + 1) / len(sorted_psnr)
    if plot:
        plt.plot(sorted_psnr, cdf, label=label)
    else:
        return cdf,sorted_psnr


def get_psnr(video_origin_path,GAI_video_path, plot=False):
    average_psnr, psnr_values = calculate_average_psnr(video_origin_path, GAI_video_path)
    if plot:
        plt.figure()
        #plot_psnr_cdf(psnr_values, 'fast')
        plot_psnr_cdf(psnr_values, 'psnr_values')
        plt.title('PSNR CDF')
        plt.xlabel('PSNR [dB]')
        plt.ylabel('CDF')
        plt.legend()
        plt.grid(True)
        plt.show()
    return average_psnr,psnr_values

####################### Image processing tools ####################### 
#  going into ran processing
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def get_other_ue(RAN_folder,rnti_ue1):
    dirs = os.listdir(RAN_folder)
    for dir_ in dirs:
        if dir_.split("_")[0]!= rnti_ue1:
            return  dir_.split("_")[0]

def get_ran_data(experiment_folder):
    RAN_folder = f'{experiment_folder}RAN/'
    file = open(f"{experiment_folder}/ue1/metadata.pickle",'rb')
    metadata = pickle.load(file)
    rnti_ue1 = metadata['rnti']
    rnti_ue2 = get_other_ue(RAN_folder,rnti_ue1)
    metrics = ['counter','ts','dl_aggr_tbs','ul_aggr_tbs','pucch_snr','pusch_snr','ul_mcs1']
    RAN_data = {ue :{} for ue in ['ue1','ue2']}
    for metric in metrics:
        try:
            file = open(f"{RAN_folder}{rnti_ue1}_{metric}.pickle",'rb')
            RAN_data['ue1'][metric] = pickle.load(file)
            file = open(f"{RAN_folder}{rnti_ue2}_{metric}.pickle",'rb')
            RAN_data['ue2'][metric] = pickle.load(file)
        except Exception as e:
            print(e)
            continue
    return RAN_data

def get_ul_data(experiment_folder,ue='ue1',plot=False):
    RAN_data = get_ran_data(experiment_folder)
    ul_diff = np.diff(RAN_data[ue]['ul_aggr_tbs'][:])
    ts = RAN_data[ue]['ts'][1:]
    if plot:
        plt.plot(ul_diff)
    return ul_diff,ts

def get_dl_data(experiment_folder,ue='ue2',plot=False):
    RAN_data = get_ran_data(experiment_folder)
    dl_diff = np.diff(RAN_data[ue]['dl_aggr_tbs'][:])
    ts = RAN_data[ue]['ts'][1:]
    if plot:
        plt.plot(dl_diff)
    return dl_diff,ts


def get_channel(experiment_folder,plot=False):
    avg_w = 500
    RAN_data = get_ran_data(experiment_folder)
    if plot:
        plt.plot(moving_average(RAN_data['ue1']['pusch_snr'],avg_w))
    RAN_data['ue1']['ts'] = RAN_data['ue1']['ts'][avg_w-1:]
    return moving_average(RAN_data['ue1']['pusch_snr'],avg_w), RAN_data['ue1']['ts']

# latency
def get_end_to_end_latency(ue1_file,ue2_file):
    file = open(f"{ue1_file}/send_frame_ts_ue1.pickle",'rb')
    ue1 = pickle.load(file)
    file = open(f"{ue2_file}/receive_frame_ts_ue2.pickle",'rb')
    ue2 = pickle.load(file)
    ue1_t = list(ue1.values())
    ue2_t = list(ue2.values())
    e2e = (np.array(ue2_t)-np.array(ue1_t))/1000000
    return e2e, ue1_t, ue2_t

def get_cdf_info(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)
    return cdf,sorted_data

# get latency

def get_results_for_experiment(experiment_folder):
    print(experiment_folder)
    exp_keys = list(os.listdir(experiment_folder))
    experiments = {key: list(os.listdir(f'{experiment_folder}/{key}')) for key in exp_keys}
    experiments['ue1'] = [i for i in experiments['ue1'] if os.path.isdir(f"{experiment_folder}/ue1/{i}")]

    latencies = {key:[] for key in experiments['ue1']}
    avg_latencies = {key:[] for key in experiments['ue1']}
    t_send_ts = {key:[] for key in experiments['ue1']}
    t_receive_ts = {key:[] for key in experiments['ue1']}
    psnrs = {key:[] for key in experiments['ue1']}
    avg_psnrs = {key:[] for key in experiments['ue1']}
    ue1_uplinkdata = {key:[] for key in experiments['ue1']}
    ue1_channel= {key:[] for key in experiments['ue1']}
    ue2_downlinkdata = {key:[] for key in experiments['ue1']}
    meas_ts = {key:[] for key in experiments['ue1']}

    for i in range(len(experiments['mec'])):
        idn = experiments['mec'][i]
        ue1 =  f"{experiment_folder}/ue1/{experiments['ue1'][i]}"
        ue2 =  f"{experiment_folder}/ue2/{experiments['ue2'][i]}"
        latencies[idn] ,t_send_ts[idn],t_receive_ts[idn]  = get_end_to_end_latency(ue1,ue2)
        avg_latencies[idn] = np.mean(latencies[idn])


    if get_RAN_data_:
        print("getting RAN data")
        data_channel,ts_channel = get_channel(experiment_folder,plot=False)
        dl_data,ul_data = [],[]
        channel = pd.DataFrame({'data':data_channel,'ts':ts_channel})
        ts = pd.DataFrame({'data':ts_channel,'ts':ts_channel})

        for i in range(len(ue1_channel)):
            idn = experiments['mec'][i]
            period_start, period_end = t_send_ts[idn][0], t_receive_ts[idn][-1]
            for output,param in zip([ue1_channel,meas_ts],[channel,ts]):
                j = param[param.ts > period_start]
                output[idn] = j[j.ts<period_end].data.values # only saving the actual data, not ts
        

    file = open(f"{experiment_folder}/ue1/metadata.pickle",'rb')
    metadata = pickle.load(file)
    videos = metadata['videos']
    videos = ["big.mp4"]*len(videos)
    for i in range(len(experiments['mec'])):
        idn = experiments['mec'][i]
        if "sent_" not in videos[i]:
            sent_video = f"sent_videos/sent_{videos[i]}"
        else:
            sent_video = f"sent_videos/{videos[i]}"
        GAI_video = f"{experiment_folder}ue2/1/received_video.mp4"
        print(sent_video,GAI_video)
        avg_psnrs[idn] ,psnrs[idn] = get_psnr(sent_video,GAI_video)


    latencies_, psnrs_ = [],[]
    ul_data,ul_channel,dl_data = [],[],[]
    for i in range(len(experiments['mec'])):
        idn = str(i+1)
        latencies_ = np.hstack([latencies_,latencies[idn]])
        psnrs_ = np.hstack([psnrs_,psnrs[idn]])
        #ul_data = np.hstack([ul_data,ue1_uplinkdata[idn]])
        ul_channel = np.hstack([ul_channel,ue1_channel[idn]])
        #dl_data = np.hstack([dl_data,ue2_downlinkdata[idn]])
    latencies = [latencies[key] for key in latencies]
    psnrs = [psnrs[key] for key in psnrs]
    ue1_uplinkdata = [ue1_uplinkdata[key] for key in ue1_uplinkdata]
    ue1_channel = [ue1_channel[key] for key in ue1_channel]
    ue2_downlinkdata = [ue2_downlinkdata[key] for key in ue2_downlinkdata]
    meas_ts = [meas_ts[key] for key in meas_ts]
    t_receive_ts = [t_receive_ts[key] for key in t_receive_ts]

    return latencies,psnrs,ue1_uplinkdata,ue1_channel,ue2_downlinkdata, meas_ts,t_receive_ts      
    return latencies_,psnrs_,ul_data,ul_channel,dl_data




# %%




experiment_folders = [
                      'newincreased/newincreased_tradscale/',
                      'newincreased/newincreased_gaiscale/'] 

get_RAN_data_ = True 

fig, ax = plt.subplots(1,2,figsize=(16,8))
output_data = {folder:[] for folder in experiment_folders}

for experiment_folder in experiment_folders:
    output_data[experiment_folder] = get_results_for_experiment(experiment_folder)
    latencies,psnrs,ul_data,ul_channel,dl_data,meas_ts,rec_ts = output_data[experiment_folder]
    latencies = np.hstack(latencies)
    psnrs= np.hstack(psnrs)
    ul_data=np.hstack(ul_data)
    ul_channel=np.hstack(ul_channel)
    dl_data=np.hstack(dl_data)
    meas_ts =np.hstack(meas_ts)
    
    cdf, psnr = get_cdf_info(np.hstack(psnrs))
    cdf_lat, latency = get_cdf_info(np.hstack(latencies))
    ax[0].plot(psnr,cdf,label = experiment_folder)

    ax[1].plot(latency,cdf_lat,label = experiment_folder)
    ax[1].set_xlim([0,150])


ax[0].title.set_text('psnr')
ax[0].set_ylabel("CDF")
ax[0].set_xlabel("PSNR")
ax[0].legend(experiment_folders)

ax[1].title.set_text('E2E latency')
ax[1].set_ylabel("CDF")
ax[1].set_xlabel("video frame latency (ms)")
ax[1].legend(experiment_folders)


# %%


save_fig = True
plt_index = 0
d_rate = 10
for experiment_folder in experiment_folders:
    latencies,psnrs,ul_data,ul_channel,dl_data,meas_ts,rec_ts = output_data[experiment_folder]
    if plt_index != -1:
        latencies = latencies[plt_index]#np.hstack(latencies)
        psnrs= psnrs[plt_index]#np.hstack(psnrs)
        ul_data= ul_data[plt_index]# np.hstack(ul_data)
        ul_channel= ul_channel[plt_index] #np.hstack(ul_channel)
        dl_data= dl_data[plt_index]# np.hstack(dl_data)
        meas_ts = meas_ts[plt_index] #np.hstack(meas_ts)
    else:
        latencies = np.hstack(latencies)
        psnrs= np.hstack(psnrs)
        ul_data= np.hstack(ul_data)
        ul_channel= np.hstack(ul_channel)
        dl_data= np.hstack(dl_data)
        meas_ts =np.hstack(meas_ts)
    
    cdf, psnr = get_cdf_info(np.hstack(psnrs))
    cdf_lat, latency = get_cdf_info(np.hstack(latencies))
    plt.plot(latency[::d_rate],cdf_lat[::d_rate],label = experiment_folder)


plt.legend()
plt.xlabel('Time')
plt.ylabel("PSNR and Channel SNR")
plt.ylim([0,1.001])
plt.xlim([15,90])
plt.title(experiment_folder)
if save_fig:
    save_folder = 'results'
    try:
        os.makedirs(f'Figures/{save_folder}/tikz/')
    except:
        pass
    save_name = 'latencycdf'
    plt.savefig(f"Figures/{save_folder}/{save_name}")
    tikz.save(f'Figures/{save_folder}/tikz/{save_name}.tex')





# %% PSNR figures

def plot_subset(rec_ts,meas_ts,index=-1,start_index = 0):
    new_rec_ = rec_ts[start_index:index] - rec_ts[start_index]

    new_meas_1 = meas_ts>=rec_ts[start_index] 
    meas_start = len(meas_ts) - sum(new_meas_1)
    new_meas_2 = meas_ts<rec_ts[index]

    new_meas_indexes = np.logical_and(np.array(new_meas_1), np.array(new_meas_2))
    return new_meas_indexes, new_rec_,meas_start
    
t_offset = 0
index_offset = 1400
factor = 5
save_fig = False
psnr_label = ['GAI','Trad']
phy_label = ['GAI_channel','Trad_channel']
plt_index = 0
d_rate = 20
for i in range(len(experiment_folders[:3])):
    experiment_folder = experiment_folders[i]
    latencies,psnrs,ul_data,ul_channel,dl_data,meas_ts,rec_ts = output_data[experiment_folder]
    start_index = 0+index_offset*factor
    index = index_offset*(factor+1)
    #start_index,index = 10050, 11000
    #start_index,index = 4800, 5650
    plt_index =0
    rec_ts= np.array(rec_ts)
    new_meas_indexes,new_rec,meas_start = plot_subset(np.array(rec_ts[plt_index]),np.array(meas_ts[plt_index]),index,start_index)
    new_meas = meas_ts[plt_index][new_meas_indexes] - meas_ts[plt_index][new_meas_indexes][0]
    plt.plot(new_rec/1000000000,psnrs[plt_index][start_index:index],label=psnr_label[i]+experiment_folder)
    #plt.plot(new_rec/1000000000,latencies[plt_index][start_index:index],label='latency')
    plt.plot(new_meas[::d_rate]/1000000000,ul_channel[plt_index][meas_start:len(new_meas)+meas_start:d_rate],label=phy_label[i]+experiment_folder)
    
    ts_video = pd.DataFrame(new_rec/1000000000)
    video = pd.DataFrame(psnrs[plt_index][start_index:index])
    ts_channel = pd.DataFrame(new_meas[::d_rate]/1000000000)
    channel = pd.DataFrame(ul_channel[plt_index][meas_start:len(new_meas)+meas_start:d_rate])



plt.legend()
plt.xlabel('Time')
plt.ylabel("PSNR and Channel SNR")
plt.ylim([24,34])
plt.title(experiment_folder)
plt.hlines(28,0,70)
if save_fig:
    save_folder = 'results'
    try:
        os.makedirs(f'Figures/{save_folder}/tikz/')
    except:
        pass
    save_name = 'SNR_timeview'
    print(f"Figures/{save_folder}/{save_name}")
    plt.savefig(f"Figures/{save_folder}/{save_name}")
    tikz.save(f'Figures/{save_folder}/tikz/{save_name}.tex')




# %% Latency figure
plt_index = 1
psnr_label = ['No GAI', 'GAI']
counter = 0
for experiment_folder in experiment_folders:
    latencies,psnrs,ul_data,ul_channel,dl_data,meas_ts,rec_ts = output_data[experiment_folder]
    if plt_index != -1:
        latencies = latencies[plt_index]#np.hstack(latencies)
        psnrs= psnrs[plt_index]#np.hstack(psnrs)
        ul_data= ul_data[plt_index]# np.hstack(ul_data)
        ul_channel= ul_channel[plt_index] #np.hstack(ul_channel)
        dl_data= dl_data[plt_index]# np.hstack(dl_data)
        meas_ts = meas_ts[plt_index] #np.hstack(meas_ts)
    else:
        latencies = latencies#np.hstack(latencies)
        psnrs= psnrs#np.hstack(psnrs)
        ul_data= ul_data# np.hstack(ul_data)
        ul_channel= ul_channel #np.hstack(ul_channel)
        dl_data= dl_data# np.hstack(dl_data)
        meas_ts = meas_ts #np.hstack(meas_ts)  
    
    cdf, psnr = get_cdf_info(np.hstack(psnrs))
    cdf_lat, latency = get_cdf_info(np.hstack(latencies))
    #fig.suptitle(experiment_folder)

    plt.plot(latency,cdf_lat,label=psnr_label[counter])
    counter+=1
plt.legend()
plt.xlim([15,100])
plt.xlabel("Latency")
plt.ylabel("CDF")
save_folder = 'results'
try:
    os.makedirs(f'Figures/{save_folder}/tikz/')
except:
    pass
save_name = 'latencycdf'
fig.savefig(f"Figures/{save_folder}/{save_name}")
tikz.save(f'Figures/{save_folder}/tikz/{save_name}.tex')



# %% processing difference figure
plt_index = -1
latencies_1,psnrs,ul_data,ul_channel,dl_data,meas_ts,rec_ts = output_data['channel/channel_gai/']
latencies_2,psnrs,ul_data,ul_channel,dl_data,meas_ts,rec_ts = output_data['channel/channel_nogai/']
if plt_index != -1:
    latencies_1 = latencies_1[plt_index]#np.hstack(latencies)
    latencies_2 = latencies_2[plt_index]#np.hstack(latencies)
else:
    latencies_1 = latencies_1
    latencies_2 = latencies_2

cdf_lat_1, latency_1 = get_cdf_info(np.hstack(latencies_1))
cdf_lat_2, latency_2 = get_cdf_info(np.hstack(latencies_2))

cutoff = 100
plt.plot(cdf_lat_1[:-cutoff],latency_2[:-cutoff]-latency_1[:-cutoff])

# %%
latencies,psnrs,ul_data,ul_channel,dl_data,meas_ts,rec_ts = output_data[experiment_folder]

plt.plot((np.array(rec_ts[0][:])-rec_ts[0][0])/1000000000,latencies[0])
plt.plot((np.array(meas_ts[0][:])-meas_ts[0][0])/1000000000,ul_channel[0])

# %%
# loading data to validate correctness
