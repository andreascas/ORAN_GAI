# ORAN_GAI
The code in this repository was used to send data through our system and apply upscaling mechanisms.\
The setup also implements other opensource solutions, which will not be delved too far into, but an overview or these are presented.\

## System dependencies
python3.10.15\
opencv-python\
pytorch - 2.1.2\
pytorch-cuda 11.8\
tensorflow 2.17.0\
tensoraudio 2.1.2\
torchvision 0.16.2\
numpy\
pandas - for analysis\
basicsr\
tikzplotlib - for plotting purposes only\
vmaf\



### RAN
We use the [OaI project](https://gitlab.eurecom.fr/oai/openairinterface5g) for the base station. From this we deploy the 4G code for an eNB. \
For the RIC we apply the [FlexRIC project](https://gitlab.eurecom.fr/mosaic5g/flexric), which provides monitoring service models, which are used to observe the channel quality.\

The RIC code must be installed on any computer intending to run the mec.py, as it utilizes SMs during the execution.

### UE
We applied the [srsRAN project](https://github.com/srsran), where we modified part of the UE code to alter its TX power over time.\
These act to transfer data from transmitter and receiver of the data is forward through the RAN.

The altering of TX power over time of UE1 is used as a feature to start our script 

### CN
We applied the [Open5gs project](https://github.com/open5gs), but any CN should work if connected to the eNB correctly.\
Version is not so relevant, as it simply acts as a gateway and endpoint for the RAN to allow the UEs to connect with the outside world.


### Flexric
We apply the [Flexric project](https://gitlab.eurecom.fr/mosaic5g/flexric), where the setup must be compiled locally
The files xapp_sdk.py and _xapp_sdk.so in /build/example/xapp/python must be copied to the /util/ folder

### Data
We used an assortment of videos from reference [18] in the article. For our convinience we made a single large video to simplify the process.


## GAI and control channel based execution

### mec.py
This script is run on the MEC with a GPU. Most of the dependencies listed were for running this script.\

Its purpose in the setup is to receive the video frames from UE1, and forward these på UE2. This also means the MEC computer must be routable for the UEs.\
It has an xApp functionality integrated which monitors the SNR of UE1, and commands it when to compress and when not to compress data.\

The required python bindings files to run xApps, xapp_sdk.py and _xapp_sdk.so must be moved to the util folder after building the flexric project.\

When applying GAI, it was run with the command:\
`$ python3 mec.py -li <mec_ip> -ue2 <ue2_ip> -ue1 <ue1_ip>  -p <port>`
 

When applying the traditional scaling we run it as:\
`$ python3 mec.py -li <mec_ip>  -ue2 <ue2_ip> -ue1 <ue1_ip> -lf nogai  -p <port>`


### ue1.py

This script was run on UE1 with the task of reading video frames, and sending these to the MEC.\
A control channel also exsist between UE1 and the MEC. According to the messages UE1 changes how it sends video frames meant for UE2.\

An aggregate video was made of all the subvideos mentioned from the article [Malicious or Benign? Towards Effective Content Moderation for Children's Videos](https://arxiv.org/abs/2305.15551).\
This is the video sent through the system.


When applying GAI\
`$ python3 ue1.py -li <ue1_ip>  -mec <mec_ip>  -p <port>`

When not applying GAI\
`$ python3 ue1.py -li <ue1_ip>  -mec <mec_ip>  -g 1 -p  <port>  --nogai`



### ue2.py
This script has the purpose of receiving VR frames and saving these by writing to a file.\
When not applying GAI, upscaling was done before writing.\

When applying GAI\
`$ python3 ue2.py -li <ue2_ip> -mec <mec_ip> -p <port>`


When not applying GAI\
`$ python3 ue2.py -li <ue2_ip>  -mec <mec_ip>  -p <port> --nogai`

## ABR-based

Similar to the GAI and semantic-based this can be run with TCP and UDP. However, in this case it most be hardcoded in the "abrprobe.py" file
Same applies to the port used for communication.

All the files impor tthe "abrprobe.py" which probes and adapts the rate in two levels for the comparison with GAI-based.
This is extendable to more high-resolution options, but for the case of the upscaling model the rate and such is configured to fit such that the low-SNR period is insignicant to support the "full" video quality, while the high-SNR can do this
The FPS and video quality are configured to support this in our current setup, but more high-resolution choices could also be adapted.

### MEC

`python3 abrmec.py`

### UE1

`python3 abrue1.py`

### UE2

`python3 abrue2.py`

## analysis.py

in the file the parameter experiment_folders must be set, which can contain the relative path to the experiment folders.\
These folders are the resulting folder after running mec.py, which collects experiment folders from ue1 and ue2 after the experiments are run.

### Vmaf analysis

Having generated the video files we generate the vmaf analysis by running.
Must place correct path to reference vidoe, and the ABR,GAI and noGAI videos to get results.

`python3 vmaf_analysis/vmafanalysis.py`