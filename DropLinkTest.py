from email.policy import default
from socket import timeout
from tokenize import Name
import gns3fy
from itertools import cycle, starmap
import time
import sys
import telnetlib
import argparse
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from InfotoGNS3 import Configurator
from multiprocessing import Pool
from functools import partial
import requests
import random
import shutil
from pcap_processor import pcap_processor
import subprocess
from ImagetoGNS3 import main as ImagetoGNS3Main

drop_num = 2

def parse():
    """
    This function parses the command-line flags

    Parameters: 
      None
    Returns:
      parser.parse_args object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', dest="host", type=str,
                        help='GNS3 server host address', default="localhost")
    parser.add_argument('-p','--port', dest="port", type=str,
                        help='GNS3 server port number', default="3080")
    # User supplies project name or a project id. 
    parser.add_argument('-n','--project-name', dest="name", type=str, 
                        help='GNS3 project name')
    parser.add_argument('-i','--project-id', dest="id", type=str,
                        help='GNS3 project id')
    parser.add_argument('-f','--project-files-path', dest='pp_path', type=str, help='GNS3 project-files path', 
                        default='../project-files')
    # TODO: Support more user defined parameters.
    parser.add_argument('-a','--addtional', dest="additional", type=str,
                        help='Additional test argument', default="")
    parser.add_argument('-o','--outputDir', dest="out", type=str,
                        help='Output Directory', default="./")
    parser.add_argument('-nt','--notes', dest="notes", type=str,
                        help='Notes', default="")
    parser.add_argument('-m','--mode', dest="mode", type=str,
                        help='Execution Mode', default="")
    parser.add_argument('--size', dest='size', type = str, 
                          default='all', help="ospf area size")
    parser.add_argument('--auto-sum', dest='auto_sum', type = str, 
                          default='N', help="eigrp auto summarization")

    return parser.parse_args()


def process_project_files(pp_path):
    '''
    Function returns a set of all assigned IP addresses and a dictionary mapping devices to IP addresses.
    The returned value follows the same format as in the InfoToGNS3.py file.
    '''
    ip_set, ip_assignment = set(), dict()
    # First reads the vpcs.(Very Easy)
    for vpcs_id in os.listdir(f'{pp_path}/vpcs'):
        fp = open(f'{pp_path}/vpcs/{vpcs_id}/startup.vpc', 'r')
        vpcs_config = fp.read()
        vpcs_ip = re.findall(r'(?<=ip ).*?(?= )', vpcs_config)[0]
        ip_set.add(vpcs_ip)
        ip_assignment[vpcs_id] = vpcs_ip
        fp.close()
    # Then reads the dynamips. (Considerably harder.)

    for dynamips_id in os.listdir(f'{pp_path}/dynamips'):
        config_path = f'{pp_path}/dynamips/{dynamips_id}/configs'
        if not os.path.isdir(config_path):
            continue
        for file_name in os.listdir(f'{pp_path}/dynamips/{dynamips_id}/configs'):
            if 'private' in file_name:
                fp = open(f'{pp_path}/dynamips/{dynamips_id}/configs/{file_name}', 'r')
                dynamips_config = fp.read()
                dynamips_interfaces = re.findall(r'(?<=FastEthernet).*?(?=\n)', dynamips_config)
                #NOTE: Assumption made. Netmask has to begin with 2 for this regex matching to work. 
                dynamips_ips = re.findall(r'(?<=address).*?(?= 2)', dynamips_config)
                for intf, ip in zip(dynamips_interfaces, dynamips_ips):
                    adaptor = re.findall(r'.*(?=\/)', intf)[0]
                    port = re.findall(r'(?<=\/).*', intf)[0]
                    ip_assignment[(dynamips_id, int(adaptor), int(port))] = ip
                    ip_set.add(ip)

    return ip_set, ip_assignment


def create_link(link, host, port):
    addr = f'http://{host}:{port}/v2/projects/{link.project_id}/links/{link.link_id}'
    r = requests.post(addr, data={'filters': link.filters, 'nodes': link.nodes})
    print(r)
    return


def random_link_drop(configurator, num, links):
    selected = []
    links_to_recover = []
    while len(selected) < num:
        rand = random.randint(0, len(links) - 1)
        while rand in selected:
            rand = random.randint(0, len(links) - 1)
        if configurator.node_dict[links[rand].nodes[0]['node_id']][1] ==  configurator.node_dict[links[rand].nodes[1]['node_id']][1] == 'dynamips':
            selected.append(rand)
    
    print("==============================================")
    for i in selected:
        links_to_recover.append(links[i])
        print(f"deleting link with link between {configurator.node_dict[links[i].nodes[0]['node_id']]}, {configurator.node_dict[links[i].nodes[1]['node_id']]}")
        links[i].delete()
    print("==============================================")
    return links_to_recover


def capture_links(configurator, lab, host, port, capture_dir):
    if not os.path.exists(capture_dir):
        os.mkdir(capture_dir, 0o777)
    for filename in os.listdir(capture_dir):
        file_path = os.path.join(capture_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    for link in lab.links:
        addr = f'http://{host}:{port}/v2/projects/{link.project_id}/links/{link.link_id}/start_capture'
        r = requests.post(addr, data={})
        # print("==============================================")
        # print(f"starting link between {configurator.node_dict[link.nodes[0]['node_id']][0]}, {configurator.node_dict[link.nodes[1]['node_id']][0]} returns status code {r}")
    time.sleep(120)
    for link in lab.links:
        addr = f'http://{host}:{port}/v2/projects/{link.project_id}/links/{link.link_id}/stop_capture'
        r = requests.post(addr, data={})
        # print("==============================================")
        # print(f"terminating link between {configurator.node_dict[link.nodes[0]['node_id']][0]}, {configurator.node_dict[link.nodes[1]['node_id']][0]} returns status code {r}")
    return


def drop_link_test(ip_assignment, HOST, PORT, OUT, PP_PATH, NAME, SIZE, AUTO_SUM, MODE):
    boot_time = time.time()
    from datetime import datetime
    print(datetime.now())
    device_level_ip_assignment = {}
    for key in ip_assignment:
        if isinstance(key, str):
            device_level_ip_assignment[ip_assignment[key].strip(' ')] = key
        elif isinstance(key, tuple):
            device_level_ip_assignment[ip_assignment[key].strip(' ')] = key[0]
    
    if not HOST or not PORT:
        # TODO: Handle more garbage conditions.
        return
    # Define the server object to establish the connection
    gns_path = PP_PATH.split("/project-files")[0]
    configurator = Configurator(f"{gns_path}/{NAME}.gns3", OUT, {'metadata'}, SIZE, AUTO_SUM)
    gns3_server = gns3fy.Gns3Connector(f"http://{HOST}:{PORT}")
    # Define the lab you want to load and assign the server connector
    lab = gns3fy.Project(name=NAME, connector=gns3_server)
    global_lab = lab
    # Retrieve its information and display
    lab.get()
    print(lab)
    print("\n")

    # Access the project attributes
    print(f"Name: {lab.name} -- Status: {lab.status} -- Is auto_closed?: {lab.auto_close}")
    #"Name: API_TEST -- Status: closed -- Is auto_closed?: False"
    print("\n")

    # Open the project
    # lab.open()
    print('I need to sleep for some time!')
    # time.sleep(80)
    print('I woke up!')

    # it works but we need to add the deleted link back to files.
    links_to_recover = random_link_drop(configurator, drop_num, lab.links)
    time.sleep(4)

    # link capture
    print("start captures!")
    capture_links(configurator, lab, HOST, PORT, PP_PATH + '/captures')

    status = pcap_processor(PP_PATH + '/captures/', MODE)
    return status


def __main__():
    args = parse()
    HOST = args.host
    global_host = HOST
    PORT = args.port
    global_port = PORT
    NAME = args.name
    global_name = NAME
    PP_PATH = args.pp_path
    global_pp_path = PP_PATH
    NOTES = args.notes
    global_notes = NOTES
    ip_set, ip_assignment = process_project_files(PP_PATH)
    global_ip_set, global_ip_assignment = ip_set, ip_assignment
    ADD = args.additional
    global_add = ADD
    SIZE = args.size
    MODE = args.mode
    AUTO_SUM = args.auto_sum 
    OUT = args.out
    

    mat_path = f'./HugeTest/{NAME}/{NAME}.txt'
    lis_path = f'./HugeTest/{NAME}/{NAME}.json'
    status = 0
    while status == 0:
        project_id = ImagetoGNS3Main(input_mode=True, 
        name = NAME, additional = MODE, outputDir = OUT, size = SIZE, auto_sum = AUTO_SUM, 
        mat = mat_path, list = lis_path)
        print(project_id)
        # open up gns3 from here.
        print(f'{OUT}{NAME}.gns3')
        proc1 = subprocess.Popen(['gns3', f'{OUT}{NAME}.gns3'])
        # Wait for a minute for GNS3 gui to boot up
        time.sleep(20)
        
        status = drop_link_test(ip_assignment, HOST, PORT, OUT, PP_PATH, NAME, SIZE, AUTO_SUM, MODE)
        time.sleep(3)
        proc1.kill()
    
    print("Process completed!")

__main__()