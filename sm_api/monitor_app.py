from re import I
import time
from datetime import datetime
import json
import os
import pickle

from util import xapp_sdk as ric
from collections import defaultdict, deque
class monitor():
    def __init__(self):
        #super().__init__()
        #ric.init()
        self.data_storage = 1000
        self.safety = 2
        self.conn = ric.conn_e2_nodes()
        self.bigdir = {}
        self.bigdir_keys = ['dl_aggr_tbs','ul_aggr_tbs','wb_cqi','dl_mcs1','ul_mcs1','pusch_snr','pucch_snr','clock','ts','counter']
        self.bigdir_keys = ['pusch_snr','pucch_snr','clock','ts','counter']

        self.mac_cb = MACCallback(self.bigdir,self.bigdir_keys)

        for i in range(0, len(self.conn)):
            print("mac handle!")
            self.MAC_handle = ric.report_mac_sm(self.conn[i].id, ric.Interval_ms_1, self.mac_cb)
            print("started")
            
    def shutdown(self,logfolder=''):
        time.sleep(1)
        ric.rm_report_mac_sm(self.MAC_handle)
        if logfolder !='':
            try:
                newdir = f"{logfolder}/RAN/"
                os.makedirs(newdir)
            except:
                print(newdir, "already exists")
                pass
            for rnti in self.bigdir:
                for key in self.bigdir[rnti]:
                    #if key != 'clock':
                        with open(f"{logfolder}/RAN/{rnti}_{key}.pickle", 'wb') as handle:
                            pickle.dump(self.bigdir[rnti][key], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        

def update_mac_index (ind, i): # TODO move into the monitor_app
    index = {}
    #index['dl_aggr_tbs'] = ind.ue_stats[i].dl_aggr_tbs  
    #index['ul_aggr_tbs'] = ind.ue_stats[i].ul_aggr_tbs 
    #index['wb_cqi'] = ind.ue_stats[i].wb_cqi
    #index['dl_mcs1'] = ind.ue_stats[i].dl_mcs1
    #index['ul_mcs1'] = ind.ue_stats[i].ul_mcs1
    index['pusch_snr']         =ind.ue_stats[i].pusch_snr          #Set?
    index['pucch_snr']         =ind.ue_stats[i].pucch_snr          #Set?

    return index


def init_ue_mac_stat_mem(mem_size):
    metrics = ['counter','ts','dl_aggr_tbs','ul_aggr_tbs','wb_cqi','dl_mcs1','ul_mcs1', 'ul_mcs2','dl_mcs2','clock','dl_aggr_prb','counter','dl_aggr_bytes_sdus','dl_aggr_sdus','pusch_snr','pucch_snr']
    mem = {metric : deque(maxlen=mem_size) for metric in metrics}
    #mem = {metric : [] for metric in metrics}
    return mem



def update_list(_list, input,counter):
    
    for index in input.keys():
        _list[str(index)].append(input[str(index)])
    _list['counter'].append(counter)
    _list['ts'].append(time.time_ns())


def init_big_dir(metrics):
    mem = {metric : [] for metric in metrics}

    return mem





############ Info on MAC_SM                         https://hackmd.io/@lfetman/SkgHNW90c           ################
class MACCallback(ric.mac_cb):
    def __init__(self,bigdir,bigdir_keys):
        self.stats={}
        self.counter = 0
        self.running = 1
        ric.mac_cb.__init__(self)
        self.stats = {}
        self.mem_size = 1000
        self.test = {}
        self.rntis = [] # TODO use the len(ind) and the RNTIs   in each to define this
        self.start = time.time_ns()
        print("Maccallback started")
        self.bigdir_keys = bigdir_keys
        self.bigdir = bigdir

        


    # Override C++ method: virtual void handle(swig_mac_ind_msg_t a) = 0;
    def handle(self, ind):
        if self.running == 1:
            if len(ind.ue_stats) > 0:
                self.counter = (self.counter + 1) #% self.mem_size   
                n = len(ind.ue_stats)
                temp_rntis = []
                for i in range(n):

                    rnti = ind.ue_stats[i].rnti
                    if rnti not in self.rntis:
                       self.test[str(rnti)] = init_ue_mac_stat_mem(self.mem_size)
                       self.bigdir[str(rnti)] = init_big_dir(self.bigdir_keys)
                       self.rntis.append(rnti)
                    new_data = update_mac_index(ind, i) # updating and adding the ones we care about
                    new_data['clock'] = (time.time_ns() - self.start) #nanoseconds
                    rnti = ind.ue_stats[i].rnti
                    temp_rntis.append(rnti)
                    new_data['clock'] = (time.time() - self.start) *1000 #millisecond
                    update_list(self.test[str(rnti)], new_data, self.counter) # attempt at updating dict list
                    update_list(self.bigdir[str(rnti)],new_data,self.counter)

                for rnti in self.rntis:
                    if rnti not in temp_rntis:
                        self.rntis.pop(self.rntis.index(rnti))
            else:
                self.rntis = []
