# ORAN_GAI
The code in this repository was used to send data through our system and apply upscaling mechanisms

The setup also applies other opensource solutions, which will not be delved too far into, but overview of these, is put.

## System dependencies



### RAN
We apply the OaI project, https://gitlab.eurecom.fr/oai/openairinterface5g. From this we deploy the 4G code for an eNB 
For the RIC we apply the flexRIC project, https://gitlab.eurecom.fr/mosaic5g/flexric, which provides monitoring service models, which are used to observe the channel quality


### UEs
We applied the srsRAN project,https://github.com/srsran, where we modified part of the UE code to alter its TX power over time.


### CN
We applied the Open5gs project,https://github.com/open5gs, but any CN should work


## code

### mec.py
This script is run on the MEC with a GPU. Most of the dependencies listed were for running this script.

Its purpose in the setup is to receive the video frames from UE1, and forward these på UE2.
It has an xApp functionality integrated which monitors the SNR of UE1, and commands it when to compress and when not to compress data.

When applying GAI, it was run with the command
'python3 mec.py -li 192.168.10.30 -ue2 192.168.100.2 -ue1 192.168.100.5  -p 60000'
 

When applying the traditional scaling we run it as 
'python3 mec.py -li 192.168.10.30 -ue2 192.168.100.2 -ue1 192.168.100.5 -lf nogai  -p 60000'



### ue1.py

This script was run on UE1 with the task of reading video frames, and sending these to the MEC.
A control channel also exsist between UE1 and the MEC. According to the messages UE1 changes how it sends video frames meant for UE2.

An aggregate video was made of all the subvideos mentioned from XXX. with the script XXXX.
This is the video sent through the system


When applying GAI
python3 ue1.py -li 192.168.100.5 -mec 192.168.10.30 -p 60000

When not applying GAI
python3 ue1.py -li 192.168.100.5 -mec 192.168.10.30 -g 1 -p  60000



### ue2.py
This script has the purpose of receiving VR frames and saving these by writing to a file.
When not applying GAI, upscaling was done before writing.

When applying GAI
python3 ue2.py -li 192.168.100.2 -mec 192.168.10.30 -p 60000


When not applying GAI
python3 ue2.py -li 192.168.100.2 -mec 192.168.10.30 -p 60000

