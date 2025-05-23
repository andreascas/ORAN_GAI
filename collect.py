import subprocess
import os
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


collect_metrics('192.168.100.3','192.168.100.2','test/','test/','bleh')