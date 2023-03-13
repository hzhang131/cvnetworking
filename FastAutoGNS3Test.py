from curses import meta
from email.policy import default
from socket import timeout
from tokenize import Name
from traceback import print_tb
import gns3fy
from itertools import cycle
import time
import sys
import telnetlib
import argparse
import re
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from InfotoGNS3 import Configurator
import json
from multiprocessing import Pool
from functools import partial

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
    # User either supplies a project name or a project id. 
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
    parser.add_argument('--size', dest='size', type = str, 
                          default='', help="ospf area size")
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

def graph_adjacency_matrix(sorted_device_names, matrix, name, additional_type, notes=""):
    fig, ax = plt.subplots(figsize = (15, 15))
    im = ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(sorted_device_names)))
    ax.set_yticks(np.arange(len(sorted_device_names)))

    ax.set_xticklabels(sorted_device_names)
    ax.set_yticklabels(sorted_device_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(sorted_device_names)):
        for j in range(len(sorted_device_names)):
            text = ax.text(j, i, matrix[i, j],
                        ha="center", va="center", color="w")

    ax.set_title("Average Trip Time Between Each Device Pair in Milliseconds.")
    plt.savefig(f'ping results {name} {additional_type} {notes}.png')
    return 

def graph_ospf_bar_chart(ospf_stt_path):
    fig, ax = plt.subplots(figsize = (15, 15))
    router_setup_times = {}
    max_connection_per_router = 0
    with open(ospf_stt_path, 'r') as f:
        dictionary = json.loads(f.read())
        for key in dictionary:
            strings = sorted(dictionary[key], key = lambda x: x.split(', ')[1])
            for string in strings:
                time_list = re.findall(r'(?<=time: )[0-9|\:|\.]*', string)
                preprocessed_time = time_list[0]
                time_segments = preprocessed_time.split(':')
                print(string, preprocessed_time, time_segments)
                duration_in_ms = float(time_segments[0]) * 1000 * 3600 + float(time_segments[1]) * 1000 * 60 + float(time_segments[2]) * 1000
                if key not in router_setup_times:
                    router_setup_times[key] = []
                router_setup_times[key].append(duration_in_ms)
            max_connection_per_router = max(max_connection_per_router, len(router_setup_times[key]))


    # draw it into a bar chart. 
    devices = list(router_setup_times.keys())
    y_pos = np.arange(len(devices))
    last = None
    cycol = cycle("bgrcmykw")
    for i in range(max_connection_per_router):
        batch = []
        for router in devices:
            if i >= len(router_setup_times[router]):
                batch.append(0)
            else:
                batch.append(router_setup_times[router][i])
        if last == None:
            ax.barh(y_pos, batch, color=next(cycol))
        else:
            input_batch = [max(0, batch[i] - last[i]) for i in range(len(last))]
            ax.barh(y_pos, input_batch, left = last, color=next(cycol))
        last = batch
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(devices, fontsize=12)
    ax.set_xlabel('OSPF Setup Time in Milliseconds')
    ax.set_title('OSPF Setup Time By Router')
    plt.axvline(max(batch), color='red', ls='dotted', linewidth = 2.5)
    plt.text(max(batch)+50,0,f'System Total OSPF Setup Time is {max(batch)/1000} seconds', rotation=90, color='blue')
    plt.savefig('ospf_setup_time_chart.png')
    return
    

def count_ospf_adjacencies(node_id, pp_path, configurator):
    '''
    This function count the number of neighboring link state adjacencies.
    '''
    node_dict = configurator.node_dict
    adj = configurator.adj
    conns = set()
    adj_count = 0
    for conn in adj[node_id]:
        source_port, destination, destination_port, source_adapter, destination_adapter = conn
        source, dest = (node_id, source_adapter, source_port), (destination, destination_adapter, destination_port)
        if node_dict[node_id][1] == 'dynamips' and node_dict[destination][1] == 'dynamips':
            sorted_conn = sorted([source, dest])
            sorted_conn = (sorted_conn[0][0], sorted_conn[0][1], sorted_conn[0][2], sorted_conn[1][0], sorted_conn[1][1], sorted_conn[1][2])
            if sorted_conn not in conns:
                conns.add(sorted_conn)
    
    # We also count the number of virtual links.
    for file_name in os.listdir(f'{pp_path}/dynamips/{node_id}/configs'):
        if 'private' in file_name:
            fp = open(f'{pp_path}/dynamips/{node_id}/configs/{file_name}', 'r')
            dynamips_config = fp.read()
            adj_count += len(re.findall(r'virtual-link', dynamips_config))

    adj_count += len(conns)
    return adj_count

def compile_results(additional_type, host, port, gns3_path, name, device_level_ip_assignment, notes=""):
    file_name = None
    if additional_type == 'eigrp':
        # eigrp test output process
        file_name = gns3_path+"/"+name+f" eigrp {notes} result.txt"
    elif additional_type == 'ospf':
        file_name = gns3_path+"/"+name+f" ospf {notes} result.txt"

    res = {}
    current_device = None
    # Define the server object to establish the connection
    gns3_server = gns3fy.Gns3Connector(f"http://{host}:{port}")
    # Define the lab you want to load and assign the server connector
    lab = gns3fy.Project(name=name, connector=gns3_server)
    lab.get()
    vpcs, dynamips = {}, {}
    device_ids_to_names = {}
    for node in lab.nodes:
        if node.node_type == 'vpcs':
            vpcs[node.name] = node.node_id
        elif node.node_type == 'dynamips':
            dynamips[node.name] = node.node_id
        device_ids_to_names[node.node_id] = node.name

    with open(file_name, "r") as fp:
        for _, line in enumerate(fp):
            if line != '\n': 
                # Name of the component.
                device_list = re.findall(r'(?<=)c_[0-9]+(?=:)', line)
                if device_list:
                    current_device = device_list[0]

                if current_device != None and current_device in dynamips:        
                    # dynamips metrics.
                    dest_ip_list = re.findall(r'(?<=Echos to ).*?(?=,)', line)
                    packet_number_list = re.findall(r'(?<=Sending ).*?(?=,)', line)
                    byte_number_list = re.findall(r'(?<=, ).*?(?=-byte)', line)
                    timeout_list = re.findall(r'(?<=is ).*?(?= seconds)', line)
                    success_rate_list = re.findall(r'(?<=rate is ).*?(?= percent)', line)
                    latency_list = re.findall(r'(?<=\= ).*?(?= ms)', line)
                    if dest_ip_list and packet_number_list and byte_number_list and timeout_list and success_rate_list and latency_list:
                        # For dynamips, each value holds (destination ip, number of packets, data size, 
                        # timeout value in seconds, success rate, minimum time in ms, average time in ms, maximum time in ms)
                        time_val = latency_list[0]
                        minimum_time, average_time, maximum_time = time_val.split('/')
                        minimum_time, average_time, maximum_time = int(minimum_time), int(average_time), int(maximum_time)
                        if dynamips[current_device] not in res:
                            res[dynamips[current_device]] = set()
                        res[dynamips[current_device]].add((dest_ip_list[0].strip(' '), int(packet_number_list[0]), 
                                                        int(byte_number_list[0]), int(timeout_list[0]), 
                                                        int(success_rate_list[0])/100, minimum_time, average_time, maximum_time))

                elif current_device != None and current_device in vpcs:        
                    # vpcs metrics.
                    # TODO: strip extra space characters at start or end of ip address.
                    dest_ip_list = re.findall(r'(?<=ping ).*?(?= result)', line)
                    # Number of packets.
                    packet_number_list = re.findall(r'(?<=-c )[0-9]+(?=\\r)', line)
                    # If the length of following data does not match the number of packets, then some of the packets may have timed out.
                    byte_number_list = re.findall(r'(?<=\\n)[0-9]+(?= bytes)', line)
                    ttl_list = re.findall(r'(?<=ttl=)[0-9]+(?= time)', line)
                    trip_time_list = re.findall(r'(?<=time=)[0-9|\.]+(?= ms)', line)
                    if dest_ip_list and packet_number_list and byte_number_list:
                        local_tuple_list = []
                        # For vpcs, the value fields a list of tuples.
                        # Each tuple contains (destination_ip, byte_number, whole number ttl in milliseconds, float trip time in milliseconds)
                        # If any given tuple times out or becomes unreachable, we fill in (destination_ip, byte_number, None, None)
                        for ttl, trip_time in zip(ttl_list, trip_time_list):
                            local_tuple_list.append((dest_ip_list[0].strip(' '), int(byte_number_list[0]), int(ttl), float(trip_time)))
                        number_of_packets = int(packet_number_list[0])
                        local_tuple_list = [(dest_ip_list[0].strip(' '), int(byte_number_list[0]), None, None) for _ in range(number_of_packets - len(local_tuple_list))] + local_tuple_list
                        if vpcs[current_device] not in res:
                            res[vpcs[current_device]] = []
                        res[vpcs[current_device]].append(local_tuple_list)
            else:
                break
    '''
    TODO: This measurement roughly gives us a big picture, but it is not 100% precise.
    '''
    connection_pair_dict = defaultdict(set)
    for key in res:
        if isinstance(res[key], list):
            dest_ip = None
            for local_tuples in res[key]:
                # calculate average round trip time for now.
                trip_count = 0
                trip_aggregate_time = 0
                for local_tuple in local_tuples:
                    dest_ip, byte_number, ttl_time, trip_time = local_tuple
                    if ttl_time != None and trip_time != None:
                        trip_count += 1
                        trip_aggregate_time += trip_time
                average_trip_time = int(trip_aggregate_time/trip_count) if trip_count != 0 else int(float('inf'))
                sorted_pair_names = sorted([key, device_level_ip_assignment[dest_ip]])
                connection_pair_dict[(sorted_pair_names[0], sorted_pair_names[1])].add(average_trip_time)
        else:
            for local_tuple in res[key]:
                dest_ip, average_trip_time = local_tuple[0], local_tuple[-1]
                sorted_pair_names = sorted([key, device_level_ip_assignment[dest_ip]])
                connection_pair_dict[(sorted_pair_names[0], sorted_pair_names[1])].add(average_trip_time)
            
    # average all times in the dictionary and create a 2d numpy array.
    sorted_device_names = sorted(list(set(device_level_ip_assignment.values())))
    inverse_device_name_indexing = {}
    for idx, name in enumerate(sorted_device_names):
        inverse_device_name_indexing[name] = idx
    
    matrix = np.zeros((len(sorted_device_names), len(sorted_device_names)))
    matrix = matrix
    for connection_pair in connection_pair_dict:
        device1, device2 = connection_pair
        average_trip_time = sum(connection_pair_dict[connection_pair]) / len(connection_pair_dict[connection_pair])
        matrix[inverse_device_name_indexing[device1], inverse_device_name_indexing[device2]] = round(average_trip_time, 1)
        matrix[inverse_device_name_indexing[device2], inverse_device_name_indexing[device1]] = round(average_trip_time, 1)

    actual_sorted_names = list(map(lambda x: device_ids_to_names[x], sorted_device_names))

    graph_adjacency_matrix(actual_sorted_names, matrix, name, additional_type, notes)
    print(notes, np.sum(matrix)/np.count_nonzero(matrix))
    matrix_shape = matrix.shape[0]
    print('unexpected zero time elements',  (matrix_shape * (matrix_shape - 1) - np.count_nonzero(matrix)) // 2) 
    # graph_ospf_bar_chart("../test4 ospf time.json")
    '''
    Design notes: 
    1. How do we handle timeouts?
    2. How do we visualize results?
    '''
    return

'''
TODO for Sunday: 
Merge the main function and compile_results, remove any redundancy.
These codes are poorly written, please spend some time to cleanup.
'''

def extract_info(node,
                global_host,
                global_name,
                global_notes,
                global_add,
                global_ip_assignment,
                global_record,
                global_ip_set,
                global_outputDir):
    with open(f"{global_outputDir}/{global_name} {global_add} {global_notes} result.txt", "a") as f:
        if node.node_type == 'dynamips' or node.node_type == 'vpcs':
            f.write(node.name + ":" + str(node.console))
            f.write("\n")
            #print(node.name + ":" + str(node.console))
            if node.node_type == "dynamips" :
                if global_add == 'ospf':
                    end_bytes = b"cold start"
                elif global_add == 'eigrp':
                    end_bytes = (f'{node.name}#').encode('ascii')
                else:
                    end_bytes = b'cold start' # may change
                start = ("\r\n" ).encode('ascii')
                telnetObj=telnetlib.Telnet(global_host,node.console)
                if global_add == 'eigrp':
                    telnetObj.write(start)
                garbage = telnetObj.read_until(match=end_bytes, timeout=10)
                # print("garbage in node " + node.name + ": \n")
                # print(str(garbage))
                # print("\n")
                if global_add == 'ospf':
                    telnetObj.write(start)
                    end_ospf = (node.name + "#").encode('ascii')
                    ospf_time = (telnetObj.read_until(match = end_ospf, timeout=10)).decode("ascii")
                    # print("router " + node.name + " ospf setting:\n")
                    lines = re.findall(r'00:.*?(?= on)', ospf_time)
                    global_record[node.name] = []
                    for line in lines:
                        if 'Nbr' in line:
                            time_nei = re.findall(r'[0-9]+:[0-9]+:[0-9]+.[0-9]+', line)
                            neighbor = re.findall(r'(?<=Nbr )[0-9]+.[0-9]+.[0-9]+.[0-9]+', line)
                            result = "neighbor: " + neighbor[0] + ", time: " + time_nei[0]
                            global_record[node.name].append(result)
                            # print("neighbor: ")
                            # print(neighbor[0])
                            # print("\n")
                            # print("time: ")
                            # print(time_nei[0])
                            # print("\n")
                    # print("\n")
                else:
                    telnetObj.write(start)
                    # print("router " + node.name + " eigrp setting:\n")
                    lines = re.findall(r'00:.*?(?= up: new adjacency)', str(garbage))
                    global_record[node.name] = []
                    for line in lines:
                        if 'Neighbor' in line:
                            time_nei = re.findall(r'[0-9]+:[0-9]+:[0-9]+.[0-9]+', line)
                            neighbor = re.findall(r'(?<=Neighbor )[0-9]+.[0-9]+.[0-9]+.[0-9]+', line)
                            result = "neighbor: " + neighbor[0] + ", time: " + time_nei[0]
                            global_record[node.name].append(result)
                    #         print("neighbor: ")
                    #         print(neighbor[0])
                    #         print("\n")
                    #         print("time: ")
                    #         print(time_nei[0])
                    #         print("\n")
                    # print("\n")
            else:
                telnetObj=telnetlib.Telnet(global_host,node.console)
                telnetObj.read_until(match=b"gateway", timeout=10)
    
            for ip_to_ping in global_ip_set:
                if node.node_type == 'dynamips':
                    if (node.node_id, 0, 0) in global_ip_assignment and global_ip_assignment[(node.node_id, 0, 0)] == ip_to_ping:
                        continue
                    if (node.node_id, 0, 1) in global_ip_assignment and global_ip_assignment[(node.node_id, 0, 1)] == ip_to_ping:
                        continue
                    if (node.node_id, 1, 0) in global_ip_assignment and global_ip_assignment[(node.node_id, 1, 0)] == ip_to_ping:
                        continue
                    if (node.node_id, 2, 0) in global_ip_assignment and global_ip_assignment[(node.node_id, 2, 0)] == ip_to_ping:
                        continue
                    message = ("ping " + ip_to_ping + " repeat 10\r\n" ).encode('ascii')
                    telnetObj.write(message)
                    output=str(telnetObj.read_until(match=b" ms",timeout=10))
                elif node.node_type == "vpcs":
                    if global_ip_assignment[node.node_id] == ip_to_ping:
                        continue
                    message = ("ping " + ip_to_ping + " -c 5\n" ).encode('ascii')
                    telnetObj.write(message)
                    output=str(telnetObj.read_until(match=b"xxxx",timeout=10))
                
                f.write("node " + node.name + " ping " + ip_to_ping + " result:")
                f.write(str(output))
                f.write('\n')
                # print("node " + node.name + " ping " + ip_to_ping + " result:")
                # print(str(output))
                # print('\n')
                time.sleep(1.5)
            telnetObj.close()
    return global_record

def __main__():
    boot_time = time.time()
    from datetime import datetime
    print(datetime.now())
    args = parse()
    HOST = args.host
    global_host = HOST
    PORT = args.port
    global_port = PORT
    NAME = args.name
    global_name = NAME
    PP_PATH = args.pp_path
    global_pp_path = PP_PATH
    ID = args.id
    global_id = ID
    NOTES = args.notes
    global_notes = NOTES
    ip_set, ip_assignment = process_project_files(PP_PATH)
    global_ip_set, global_ip_assignment = ip_set, ip_assignment
    device_level_ip_assignment = {}
    for key in ip_assignment:
        if isinstance(key, str):
            device_level_ip_assignment[ip_assignment[key].strip(' ')] = key
        elif isinstance(key, tuple):
            device_level_ip_assignment[ip_assignment[key].strip(' ')] = key[0]
    ADD = args.additional
    global_add = ADD
    if not HOST or not PORT:
        # TODO: Handle more garbage conditions.
        return
    # Define the server object to establish the connection
    outputDir = args.out
    global_outputDir = outputDir
    gns_path = PP_PATH.split("/project-files")[0]
    configurator = Configurator(f"{gns_path}/{NAME}.gns3", outputDir, {'metadata'}, args.size, args.auto_sum)
    gns3_server = gns3fy.Gns3Connector(f"http://{HOST}:{PORT}")

    # Define the lab you want to load and assign the server connector
    lab = gns3fy.Project(name=NAME, connector=gns3_server)
    global_lab = lab
    # Retrieve its information and display
    lab.get()
    print(lab)
    #"Project(project_id='4b21dfb3-675a-4efa-8613-2f7fb32e76fe', name='API_TEST', status='opened', ...)"
    print("\n")

    # Access the project attributes
    print(f"Name: {lab.name} -- Status: {lab.status} -- Is auto_closed?: {lab.auto_close}")
    #"Name: API_TEST -- Status: closed -- Is auto_closed?: False"
    print("\n")

    # Open the project
    # lab.open()
    print('I need to sleep for some time!')
    time.sleep(100)
    print('I woke up!')

    # Verify the stats
    print(lab.stats)
    print("\n")

    global_record = {}
    with open(f"{outputDir}/{NAME} {ADD} {NOTES} result.txt", "w") as f, open(f"{outputDir}/{NAME} {ADD} {NOTES} time.json", "w") as t:
        pass
    # Read the names and status of all the nodes in the project
    pool = Pool(16)
    global_res = pool.map(partial(extract_info, global_host=global_host,
                                   global_name=global_name,
                                   global_notes=global_notes,
                                   global_add=global_add,
                                   global_ip_assignment=global_ip_assignment,
                                   global_record=global_record,
                                   global_ip_set=global_ip_set,
                                   global_outputDir=global_outputDir), lab.nodes)
    pool.close()
    pool.join()
    with open(f"{outputDir}/{NAME} {ADD} {NOTES} time.json", "w") as t:
        t.write(json.dumps(global_res, sort_keys=True, indent=4))
    print(f'All Tests elapsed in {time.time() - boot_time}')

    compile_results(ADD, HOST, PORT, outputDir, NAME, device_level_ip_assignment, NOTES)

    ###########System Setup Time Report###########
    string = ''
    for res in global_res:
        for key in res:
            for metadata in res[key]:
                query_res = re.findall(r'(?<=time: ).*', metadata)
                if query_res:
                    string = max(string, query_res[0])

    print(f'System max converge time is {string}')
    ################Report End#################

    return
__main__()
