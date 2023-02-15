import copy
import json
from multiprocessing import connection
from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from turtle import begin_fill, end_fill
from xmlrpc.client import ProtocolError
from distutils.command import clean
from xmlrpc.server import CGIXMLRPCRequestHandler
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import collections
from networkx.algorithms.traversal.depth_first_search import dfs_tree

BACKBONE = 0
ROUTER = 0
SWITCH = 1
VPC = 4
PHONE = 5
FIREWALL = 8
INTERNET = 9
CLASS_DICT = {ROUTER: {"compute_id": "local",
                    "console": 5000,
                    "console_auto_start": False,
                    "console_type": "telnet",
                    "custom_adapters": [],
                    "first_port_name": None,
                    "height": 45,
                    "label": {
                        "rotation": 0,
                        "style": "font-family: TypeWriter;font-size: 10.0;font-weight: bold;fill: #000000;fill-opacity: 1.0;",
                        "text": "R1",
                        "x": 9,
                        "y": -25
                    },
                    "locked": False,
                    "name": "R1",
                    "node_id": "7e0a3878-58af-4437-a030-efe7de7458cf",
                    "node_type": "dynamips",
                    "port_name_format": "Ethernet{0}",
                    "port_segment_size": 0,
                    "properties": {
                        "auto_delete_disks": True,
                        "aux": None,
                        "clock_divisor": 8,
                        "disk0": 0,
                        "disk1": 0,
                        "dynamips_id": 1,
                        "exec_area": 64,
                        "idlemax": 500,
                        "idlepc": "",
                        "idlesleep": 30,
                        "image": "c3725-adventerprisek9-mz124-15.image",
                        "image_md5sum": "1c950444f3261338c3d42e72a6ded980",
                        "iomem": 5,
                        "mac_addr": "c201.6d47.0000",
                        "mmap": True,
                        "nvram": 256,
                        "platform": "c3725",
                        "ram": 128,
                        "slot0": "GT96100-FE",
                        "slot1": "NM-1FE-TX",
                        "slot2": "NM-1FE-TX",
                        "sparsemem": True,
                        "system_id": "FTX0945W0MY",
                        "usage": "",
                        "wic0": None,
                        "wic1": None,
                        "wic2": None
                    },
                    "symbol": ":/symbols/router.svg",
                    "template_id": "678961ca-7b9a-44a9-bab0-f376becf147f",
                    "width": 66,
                    "x": -303,
                    "y": -82,
                    "z": 1},
              SWITCH: {"compute_id": "local",
                    "console": 5003,
                    "console_auto_start": False,
                    "console_type": "none",
                    "custom_adapters": [],
                    "first_port_name": None,
                    "height": 32,
                    "label": {
                        "rotation": 0,
                        "style": "font-family: TypeWriter;font-size: 10.0;font-weight: bold;fill: #000000;fill-opacity: 1.0;",
                        "text": "Switch1",
                        "x": -28,
                        "y": -25
                    },
                    "locked": False,
                    "name": "Switch1",
                    "node_id": "6a9cd58e-ab66-485e-8bc2-6ce24657f16f",
                    "node_type": "ethernet_switch",
                    "port_name_format": "Ethernet{0}",
                    "port_segment_size": 0,
                    "properties": {
                        "ports_mapping": [
                            {
                                "name": "Ethernet0",
                                "port_number": 0,
                                "type": "access",
                                "vlan": 1
                            },
                            {
                                "name": "Ethernet1",
                                "port_number": 1,
                                "type": "access",
                                "vlan": 1
                            },
                            {
                                "name": "Ethernet2",
                                "port_number": 2,
                                "type": "access",
                                "vlan": 1
                            },
                            {
                                "name": "Ethernet3",
                                "port_number": 3,
                                "type": "access",
                                "vlan": 1
                            },
                            {
                                "name": "Ethernet4",
                                "port_number": 4,
                                "type": "access",
                                "vlan": 1
                            },
                            {
                                "name": "Ethernet5",
                                "port_number": 5,
                                "type": "access",
                                "vlan": 1
                            },
                            {
                                "name": "Ethernet6",
                                "port_number": 6,
                                "type": "access",
                                "vlan": 1
                            },
                            {
                                "name": "Ethernet7",
                                "port_number": 7,
                                "type": "access",
                                "vlan": 1
                            }
                        ]
                    },
                    "symbol": ":/symbols/ethernet_switch.svg",
                    "template_id": "1966b864-93e7-32d5-965f-001384eec461",
                    "width": 72,
                    "x": -418,
                    "y": -153,
                    "z": 1}, 
              VPC: {"compute_id": "local",
                    "console": 5005,
                    "console_auto_start": False,
                    "console_type": "telnet",
                    "custom_adapters": [],
                    "first_port_name": None,
                    "height": 59,
                    "label": {
                        "rotation": 0,
                        "style": "font-family: TypeWriter;font-size: 10.0;font-weight: bold;fill: #000000;fill-opacity: 1.0;",
                        "text": "PC1",
                        "x": -4,
                        "y": -46
                    },
                    "locked": False,
                    "name": "PC1",
                    "node_id": "65467f8d-119f-4726-8646-df5b7610cc20",
                    "node_type": "vpcs",
                    "port_name_format": "Ethernet{0}",
                    "port_segment_size": 0,
                    "properties": {},
                    "symbol": ":/symbols/vpcs_guest.svg",
                    "template_id": "19021f99-e36f-394d-b4a1-8aaa902ab9cc",
                    "width": 65,
                    "x": -533,
                    "y": -254,
                    "z": 1},
              PHONE: {"compute_id": "local",
                    "console": 5005,
                    "console_auto_start": False,
                    "console_type": "telnet",
                    "custom_adapters": [],
                    "first_port_name": None,
                    "height": 59,
                    "label": {
                        "rotation": 0,
                        "style": "font-family: TypeWriter;font-size: 10.0;font-weight: bold;fill: #000000;fill-opacity: 1.0;",
                        "text": "PC1",
                        "x": -4,
                        "y": -46
                    },
                    "locked": False,
                    "name": "PC1",
                    "node_id": "65467f8d-119f-4726-8646-df5b7610cc20",
                    "node_type": "vpcs",
                    "port_name_format": "Ethernet{0}",
                    "port_segment_size": 0,
                    "properties": {},
                    "symbol": ":/symbols/vpcs_guest.svg",
                    "template_id": "19021f99-e36f-394d-b4a1-8aaa902ab9cc",
                    "width": 65,
                    "x": -533,
                    "y": -254,
                    "z": 1},
              FIREWALL: {"compute_id": "local",
                        "console": 5900,
                        "console_auto_start": False,
                        "console_type": "vnc",
                        "custom_adapters": [],
                        "first_port_name": "Management0/0",
                        "height": 60,
                        "label": {
                            "rotation": 0,
                            "style": "font-family: TypeWriter;font-size: 10.0;font-weight: bold;fill: #000000;fill-opacity: 1.0;",
                            "text": "CiscoASAv9.8.1-1",
                            "x": -111,
                            "y": -25
                        },
                        "locked": False,
                        "name": "CiscoASAv9.8.1-1",
                        "node_id": "cd1f2f3d-5101-474b-9980-e653c3dd78c7",
                        "node_type": "qemu",
                        "port_name_format": "Gi0/{0}",
                        "port_segment_size": 0,
                        "properties": {
                            "adapter_type": "e1000",
                            "adapters": 8,
                            "bios_image": "",
                            "bios_image_md5sum": None,
                            "boot_priority": "c",
                            "cdrom_image": "",
                            "cdrom_image_md5sum": None,
                            "cpu_throttling": 0,
                            "cpus": 1,
                            "create_config_disk": False,
                            "hda_disk_image": "asav981.qcow2",
                            "hda_disk_image_md5sum": "8d3612fe22b1a7dec118010e17e29411",
                            "hda_disk_interface": "virtio",
                            "hdb_disk_image": "",
                            "hdb_disk_image_md5sum": None,
                            "hdb_disk_interface": "none",
                            "hdc_disk_image": "",
                            "hdc_disk_image_md5sum": None,
                            "hdc_disk_interface": "none",
                            "hdd_disk_image": "",
                            "hdd_disk_image_md5sum": None,
                            "hdd_disk_interface": "none",
                            "initrd": "",
                            "initrd_md5sum": None,
                            "kernel_command_line": "",
                            "kernel_image": "",
                            "kernel_image_md5sum": None,
                            "legacy_networking": False,
                            "linked_clone": True,
                            "mac_address": "0c:1f:2f:3d:00:00",
                            "on_close": "power_off",
                            "options": "",
                            "platform": "x86_64",
                            "process_priority": "normal",
                            "qemu_path": "/usr/bin/qemu-system-x86_64",
                            "ram": 2048,
                            "replicate_network_connection_state": True,
                            "usage": "There is no default password and enable password. A default configuration is present. ASAv goes through a double-boot before becoming active. This is normal and expected."
                        },
                        "symbol": ":/symbols/classic/asa.svg",
                        "template_id": "68f8b26c-b1ca-499d-bb9b-f6ea5b733035",
                        "width": 52,
                        "x": -382,
                        "y": 26,
                        "z": 1}}

def clean_string_ip_address(s):
    sl = s.split('.')
    sl[-1] = '0'
    return '.'.join(sl)


def set_component_config(comp_id:int, configs:dict):
    if comp_id not in CLASS_DICT:
        return
    else:
        local_configs = copy.copy(configs)
        skeleton = CLASS_DICT[comp_id]
        for i in local_configs:
            if '-' in i:
                first_lvl, second_lvl = i.split('-')[0], i.split('-')[1]
                skeleton[first_lvl][second_lvl] = local_configs[i]
            else:
                skeleton[i] = local_configs[i]
        node_x, node_y = int(local_configs['x']), int(local_configs['y'])
        label_x, label_y = int(local_configs['x']), int(local_configs['y']-20)
        skeleton['label']['x'] = 0
        skeleton['label']['y'] = -30
    
    return skeleton

"""
Top level conversion file.
Saves file to name.gns3 
"""
def generate_gns3file(name, GNS3dir, nodedicts:dict, adjacency, project_id="0e856125-d8eb-43db-93c3-27b4c4d0175e",auto_close = True, auto_open = False, auto_start = True, 
                      drawing_grid_size = 25, grid_size = 75, scene_height = 1000, scene_width = 2000, 
                      show_grid = False, show_interface_labels = False, show_layers = False, snap_to_grid = False,
                      supplier = None, variables = None, zoom = 100, revision = 9):
    topofile_skeleton = {
        "auto_close": auto_close, 
        "auto_open": auto_open,
        "auto_start": auto_start,
        "drawing_grid_size": drawing_grid_size,
        "grid_size": grid_size,
        "name": name,
        "project_id": project_id,
        "revision": revision,
        "scene_height": scene_height,
        "scene_width": scene_width,
        "show_grid": show_grid,
        "show_interface_labels": show_interface_labels,
        "show_layers": show_layers,
        "snap_to_grid": snap_to_grid,
        "supplier": supplier,
        "topology": {},
        "type": "topology", 
        "variables": variables,
        "version": "2.2.31",
        "zoom": zoom
    }
    
    # Prepares to arrange the node connections.
    topology = {
        "computes": [],
        "drawings": [],
        "links": [],
        "nodes": []
    }
    
    # node_ids, template_ids follow the order.
    coords = node_spacing(nodedicts, adjacency, scene_height, scene_width)
    # print("init", nodedicts)
    nodes = generate_nodes(nodedicts, adjacency, coords)
    # print("after generate nodes", len(nodes), "\n\n\n\n\n\n\n")
    topology["nodes"] = nodes
    links = generate_links(nodedicts, adjacency, nodes)
    topology["links"] = links
    topofile_skeleton["topology"] = topology
    with open(f'{GNS3dir}/{name}.gns3', 'w+') as f:
        f.write(json.dumps(topofile_skeleton, sort_keys=True, indent=4))
    return

# TODO: generate links.
'''
We will not connect more routers if the interfaces have already been used up!
'''

def generate_links(nodedicts:dict, adjacency, nodes):
    local_nodes = copy.deepcopy(nodes)
    PORTS_LOOKUP = {
                "vpcs": [(0, 0)],
                "qemu": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
                "dynamips": [(0,0), (0,1), (1, 0), (2, 0)],
                "ethernet_switch": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)]
    }
    skeleton = {
                "filters": {},
                "link_id": "89d7a57e-92a2-497e-8e4f-76887f4a6b80",
                "link_style": {},
                "nodes": [
                    {
                        "adapter_number": 0,
                        "label": {
                            "rotation": 0,
                            "style": "font-family: TypeWriter;font-size: 10.0;font-weight: bold;fill: #000000;fill-opacity: 1.0;",
                            "text": "f0/0",
                            "x": 0,
                            "y": 0
                        },
                        "node_id": "CFd2C722-5d95-fF11-b991-D8d9DabF7470",
                        "port_number": 0
                    },
                    {
                        "adapter_number": 0,
                        "label": {
                            "rotation": 0,
                            "style": "font-family: TypeWriter;font-size: 10.0;font-weight: bold;fill: #000000;fill-opacity: 1.0;",
                            "text": "f0/0",
                            "x": 0,
                            "y": 0
                        },
                        "node_id": "76775b8d-aB8C-d0dd-5A1b-aa57B485DDb1",
                        "port_number": 1
                    }
                ],
                "suspend": False
            }
    node_id_all_ports = {}
    links = []
    adj_list = adjacency_matrix_to_set(adjacency)
    node_infos = find_node_ids(local_nodes)
    ROUTER_TO_ROUTER_ADP_PORTS = [(0, 0), (0, 1)]
    for node in node_infos:
        node_id, node_type = node_infos[node][0], node_infos[node][1]
        node_id_all_ports[node_id] = {}
        for adp_port in PORTS_LOOKUP[node_type]:
            node_id_all_ports[node_id][adp_port] = 0
    
    for conn in adj_list:
        src, dst = conn[0], conn[1]
        # if the type is illegal, skip it. 
        if src not in node_infos or dst not in node_infos:
            print("skipped, not in nodeinfo", src, dst, src not in node_infos, dst not in node_infos)
            continue
        src_id, src_type = node_infos[src]
        dst_id, dst_type = node_infos[dst]
        # after getting the connection source and the connection destination
        # we figure out what ports to connect to
        src_first_available_adp_port, dst_first_available_adp_port = None, None
        for adp_port in node_id_all_ports[src_id]:
            if node_id_all_ports[src_id][adp_port] == 0:
                src_first_available_adp_port = adp_port
                break
        
        for adp_port in node_id_all_ports[dst_id]:
            if node_id_all_ports[dst_id][adp_port] == 0:
                dst_first_available_adp_port = adp_port
                break
        
        # drop connection if no ports available.
        if src_first_available_adp_port == None or dst_first_available_adp_port == None:
            print('dropped', src, dst, node_id_all_ports[src_id], node_id_all_ports[dst_id])
            continue
        
        # commit to these adp_port connections.
        node_id_all_ports[src_id][src_first_available_adp_port] = 1
        node_id_all_ports[dst_id][dst_first_available_adp_port] = 1
        
        # generate new link id.
        new_link_id = random_id_generator()
        
        # feed all variables into the skeleton json
        new_skeleton = copy.deepcopy(skeleton)
        new_skeleton["link_id"] = new_link_id
        new_skeleton["nodes"][0]["adapter_number"] = src_first_available_adp_port[0]
        new_skeleton["nodes"][0]["node_id"] = src_id
        new_skeleton["nodes"][0]["port_number"] = src_first_available_adp_port[1]
        
        new_skeleton["nodes"][1]["adapter_number"] = dst_first_available_adp_port[0]
        new_skeleton["nodes"][1]["node_id"] = dst_id
        new_skeleton["nodes"][1]["port_number"] = dst_first_available_adp_port[1]
        
        links.append(new_skeleton)
        
    return links

# TODO: generate nodes.
def generate_nodes(nodedicts:dict, adjacency, coords:dict) -> list:
    # Generate config dict for each node.
    # Return a list of dicts.
    dict_list = []
    for i in range(len(nodedicts)):
        comp_id = nodedicts[i]['pred_class']
        node_id = nodedicts[i]['id']
        node_x, node_y = int(coords[node_id][0]), int(coords[node_id][1])
        config = {"x": node_x, 
                  "y": node_y, 
                  "node_id": random_id_generator(), 
                  "console": 5000 + node_id, 
                  "name": f'c_{node_id}'}
        if comp_id == 0:
            config["properties-mac_addr"] = random_mac_generator(comp_id)
            config["properties-dynamips_id"] = 10 + node_id
        elif comp_id == 8:
            config["properties-mac_address"] = random_mac_generator(comp_id)
        config["label-text"] = f'C_{node_id}'
        component_config = set_component_config(comp_id, config)
        if component_config:
            dict_list.append(copy.deepcopy(component_config))
    return dict_list

def adjacency_matrix_to_set(adjacency):
    adj_set = set()
    for src, row in enumerate(adjacency):
        dst_indices = list(np.where(row==1))
        for dst in dst_indices[0]:
            if (src, dst) not in adj_set and (dst, src) not in adj_set:
                adj_set.add((src, dst))
    return adj_set

# generate a random id that conforms to standard.
def random_id_generator() -> str:
    first_8, second_4, third_4, fourth_4, fifth_12 = str(), str(), str(), str(), str()
    # 0-9A-Fa-F
    
    correspondence = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 
                      10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F",
                      16: "a", 17: "b", 18: "c", 19: "d", 20: "e", 21:"f"}
    
    first_8 = ''.join([correspondence[i] for i in np.random.randint(22, size=8)])
    second_4 = ''.join([correspondence[i] for i in np.random.randint(22, size=4)])
    third_4 = ''.join([correspondence[i] for i in np.random.randint(22, size=4)])
    fourth_4 = ''.join([correspondence[i] for i in np.random.randint(22, size=4)])
    fifth_12 = ''.join([correspondence[i] for i in np.random.randint(22, size=12)])
    
    return f'{first_8}-{second_4}-{third_4}-{fourth_4}-{fifth_12}'
    
def random_mac_generator(device: int) -> str:
    '''
    Now it ONLY supports router and Cisco ASAV firewall
    '''
    correspondence = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 
                      10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F",
                      16: "a", 17: "b", 18: "c", 19: "d", 20: "e", 21:"f"}
    if device == 8:
        first_2 = ''.join([correspondence[i] for i in np.random.randint(22, size=2)])
        second_2 = ''.join([correspondence[i] for i in np.random.randint(22, size=2)])
        third_2 = ''.join([correspondence[i] for i in np.random.randint(22, size=2)])
        fourth_2 = ''.join([correspondence[i] for i in np.random.randint(22, size=2)])
        fifth_2 = ''.join([correspondence[i] for i in np.random.randint(22, size=2)])
        sixth_2 = ''.join([correspondence[i] for i in np.random.randint(22, size=2)])
        return f'{first_2}:{second_2}:{third_2}:{fourth_2}:{fifth_2}:{sixth_2}'
    elif device == 0:
        first_4 = 'c2' + ''.join([correspondence[i] for i in np.random.randint(22, size=2)])
        second_4 = ''.join([correspondence[i] for i in np.random.randint(22, size=4)])
        third_4 = ''.join([correspondence[i] for i in np.random.randint(22, size=4)])
        return f'{first_4}.{second_4}.{third_4}'
    else:
        return "Not implmented!!!"
    
def node_spacing_old(node_dicts, adjacency_matrix, scene_height, scene_width):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    coords = nx.spring_layout(gr)
    minimum_x, minimum_y = 1000000000, 1000000000
    for node in coords:
        minimum_x = min(coords[node][0], minimum_x)
        minimum_y = min(coords[node][1], minimum_y)
    
    maximum_x, maximum_y = 0, 0
    for node in coords:
        if minimum_x < 0:
            coords[node][0] -= minimum_x
        if minimum_y < 0:
            coords[node][1] -= minimum_y
        maximum_x = max(maximum_x, coords[node][0])
        maximum_y = max(maximum_y, coords[node][0])
        
    # Normalize within 0, 1 and within canvas (2000, 1000)
    for node in coords:
        coords[node][0] /= maximum_x   # if maximum is too close to zero, 
        coords[node][1] /= maximum_y   #   normalized coordinates will be too large
        coords[node][0] *= scene_width
        coords[node][1] *= scene_height
        coords[node][0] = int(coords[node][0] - 1000)
        coords[node][1] = int(coords[node][1] - 500)
        
    # Push nodes along the border inwards.
    for node in coords:
        if coords[node][0] < 100:
            coords[node][0] += 100
        if coords[node][0] > scene_width - 100:
            coords[node][0] -= 100
            
        if coords[node][1] < 100:
            coords[node][1] += 100
        if coords[node][1] > scene_width - 100:
            coords[node][1] -= 100
    return coords

def node_spacing(node_dicts, adjacency_matrix, scene_height, scene_width):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    coords = nx.spring_layout(gr) # gives [x , y] in range [-1, 1], 
                                  #   thus we can directly multiply x and y with width/2, height/2,
                                  #   or we can leave some space for border, like use width * 0.9 / 2
        
    # change node coordinates to canvas size
    for node in coords:
        coords[node][0] *= scene_width * 0.9 / 2
        coords[node][1] *= scene_height * 0.9 / 2
        
    return coords

def find_node_ids(nodes):
    node_correspondence = {}
    for node in nodes:
        # print("looking up node in nodes", node)
        node_correspondence[int(node["name"][2:])] = (node['node_id'], node['node_type'])
    return node_correspondence

def visualize_gns3_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, with_labels=True)

def generate_router_graph(conns, node_dict) -> nx.Graph:
    G = nx.Graph()
    for s, _, _, d, _, _ in conns:
        if node_dict[s][1] == 'dynamips' and node_dict[d][1] == 'dynamips':
            G.add_edge(s, d)
    return G

'''
Deprecated: Please delete in future iterations!!!!!
'''
# def partition(G, area_size):
#     # returns a dictionary area assigned routers
#     # as well as a set of all bordering area ports.
#     nodes = np.array(G.nodes())
#     np.random.shuffle(nodes)
#     a, b = nodes[:nodes.shape[0]//2], nodes[nodes.shape[0]//2:]
#     incident_set = set()
#     res = []
#     graph_partition(G, a, b, res, area_size)
    
#     connected_subgraphs = collections.defaultdict(set)
#     node_group_assignment = {}
#     inv_node_group_assignment = collections.defaultdict(set)
#     for idx, r in enumerate(res):
#         for node in r:
#             node_group_assignment[node] = idx
#             inv_node_group_assignment[idx].add(node)
    
#     A = nx.adjacency_matrix(G)
#     A = A.todense()
#     ordered_nodes = list(G.nodes())
#     node_pos = {}
#     for idx, on in enumerate(ordered_nodes):
#         node_pos[on] = idx
#     for idx, r in enumerate(res):
#         for node in r:
#             a = np.array(A[node_pos[node]])[0]
#             incident_nodes = np.where(a == 1)
#             for incident_node in incident_nodes[0]:
#                 if node_group_assignment[node] != node_group_assignment[ordered_nodes[incident_node]] and (ordered_nodes[incident_node], node) not in incident_set:
#                     incident_set.add((node, ordered_nodes[incident_node]))
#                     connected_subgraphs[node_group_assignment[node]].add(node_group_assignment[ordered_nodes[incident_node]])
#                     connected_subgraphs[node_group_assignment[ordered_nodes[incident_node]]].add(node_group_assignment[node])

#     # rank areas based on connectivity degrees.
#     areas = []
#     for i in connected_subgraphs:
#         areas.append((i, len(connected_subgraphs[i])))
#     areas = sorted(areas, key = lambda x: (-x[1], len(inv_node_group_assignment[x[0]])))

#     area_assignment = {}
#     inv_area_assignment = {}
#     currently_assigned_area = 0
#     for a in areas:
#         index, _ = a
#         area_assignment[currently_assigned_area] = inv_node_group_assignment[index]
#         # fill in inv_area_assignment
#         for n in inv_node_group_assignment[index]:
#             inv_area_assignment[n] = currently_assigned_area
#         currently_assigned_area += 1

#     for node in inv_area_assignment:
#         print(node, inv_area_assignment[node])

#     return area_assignment, inv_area_assignment, incident_set

def most_connected_node(G):
    max_conn_degree, max_conn_node = 0, None
    adj = collections.defaultdict(set)
    for s, d in list(G.edges()):
        adj[s].add(d)
        adj[d].add(s)
    for node in adj:
        connectivity = len(adj[node])
        max_conn_node = node if connectivity > max_conn_degree else max_conn_node
        max_conn_degree = max(max_conn_degree, connectivity)
    return max_conn_node

def dfs(root, G, profile):
    # First, convert all edges under root to be directed edges.
    # Construct a new directed tree. 
    # Then we use dfs_tree from nx's algorithm package to extract all nodes under.
    adj = collections.defaultdict(set)
    children = collections.defaultdict(set)
    nG = nx.DiGraph()
    for s, d in list(G.edges()):
        adj[s].add(d)
        adj[d].add(s)
        
    queue, visited = [root], {root}
    while queue:
        key = queue.pop(0)
        for child in adj[key]:
            if child not in visited:
                children[key].add(child)
                visited.add(child)
                queue.append(child)
                nG.add_edge(key, child)

    
    for neigh in adj[root]:
        profile[neigh] = set(dfs_tree(nG, neigh).nodes())
    return

def graph_partition(G, target_size, res, allocated_nodes):
    T = nx.minimum_spanning_tree(G)
    root = most_connected_node(T)
    profile = collections.defaultdict(list)
    if target_size >= len(G.nodes()):
        res.append(set(G.nodes()))
        return set(G.nodes())
    dfs(root, T, profile)
    profile_keys = list(profile.keys())
    first_set, second_set = profile_keys[:len(profile_keys)//2], profile_keys[len(profile_keys)//2:]
    first_set_nodes, second_set_nodes = set(), set()
    for fs in first_set:
        if fs != root:
            first_set_nodes = first_set_nodes.union(set(profile[fs]))

    for ss in second_set:
        if ss != root:
            second_set_nodes = second_set_nodes.union(set(profile[ss]))
    
    # first_set_nodes, second_set_nodes = list(first_set_nodes), list(second_set_nodes)
    # if len(first_set_nodes) <= len(second_set_nodes):
    #     first_set_nodes.append(root)
    # else:
    #     second_set_nodes.append(root)
    
    first_set_nodes.add(root)
    second_set_nodes.add(root)
    
    unassigned_first_set_nodes = first_set_nodes - first_set_nodes.intersection(allocated_nodes)
    if len(unassigned_first_set_nodes) > target_size:
        SG = G.subgraph(first_set_nodes)
        allocated_nodes = graph_partition(SG, target_size, res, allocated_nodes)

    elif unassigned_first_set_nodes:
        res.append(unassigned_first_set_nodes)
        allocated_nodes = allocated_nodes.union(first_set_nodes)

    unassigned_second_set_nodes = second_set_nodes - second_set_nodes.intersection(allocated_nodes)
    if len(unassigned_second_set_nodes) > target_size:
        SG = G.subgraph(second_set_nodes)
        allocated_nodes = graph_partition(SG, target_size, res, allocated_nodes)
    elif unassigned_second_set_nodes:
        res.append(unassigned_second_set_nodes)
        allocated_nodes = allocated_nodes.union(second_set_nodes)
    return allocated_nodes

def graph_partition_wrapper(G, target_size):
    res, allocated_nodes = [], set()
    allocated_nodes = graph_partition(G, target_size, res, allocated_nodes)
    connected_subgraphs = collections.defaultdict(set)
    node_group_assignment = {}
    incident_set = set()
    # inv_node_group_assignment = collections.defaultdict(set)
    inv_node_group_assignment = {}
    for idx, r in enumerate(res):
        for node in r:
            node_group_assignment[node] = idx
            if idx not in inv_node_group_assignment:
                inv_node_group_assignment[idx] = set()
            inv_node_group_assignment[idx].add(node)

    # area_assignment, inv_area_assignment, incident_set
    node_order, inv_node_order = {}, {}
    for idx, i in enumerate(list(G.nodes())):
        node_order[idx] = i
        inv_node_order[i] = idx

    A = nx.adjacency_matrix(G)
    A = A.todense()
    for idx, r in enumerate(res):
        for node in r:
            a = np.array(A[inv_node_order[node]])[0]
            incident_nodes = np.where(a == 1)
            for incident_node in incident_nodes[0]:
                if node_group_assignment[node] != node_group_assignment[node_order[incident_node]] and (node_order[incident_node], node) not in incident_set:
                    incident_set.add((node, node_order[incident_node]))
                    connected_subgraphs[node_group_assignment[node]].add(node_group_assignment[node_order[incident_node]])
                    connected_subgraphs[node_group_assignment[node_order[incident_node]]].add(node_group_assignment[node])

    areas = []
    for i in connected_subgraphs:
        areas.append((i, len(connected_subgraphs[i])))
    if connected_subgraphs: 
        areas = sorted(areas, key = lambda x: (-x[1], len(inv_node_group_assignment[x[0]])))
    else:
        # This is reserved for the case where there is only one area in the whole topology. 
        areas = [(0, 0)]

    assignment = {}
    currently_assigned_area = 0
    for a in areas:
        index, _ = a
        assignment[index] = currently_assigned_area
        currently_assigned_area += 1

    area_assignment, inv_area_assignment = {}, {}
    for a in assignment:
        ar = assignment[a]
        area_assignment[ar] = inv_node_group_assignment[a]
        for node in inv_node_group_assignment[a]:
            if node in inv_area_assignment:
                print("ERRORRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
            inv_area_assignment[node] = ar
    
    return area_assignment, inv_area_assignment, incident_set

def convert_int_to_ip_address(integer_ip):
    range1 = integer_ip >> 24
    range2 = (integer_ip >> 16) & 0b0000000011111111 
    range3 = (integer_ip >> 8) & 0b000000000000000011111111 
    range4 = integer_ip & 0b11111111
    return f'{str(range1)}.{str(range2)}.{str(range3)}.{str(range4)}'
    
def convert_ip_address_to_int(string_ip):
    string_ip_list = string_ip.split('.')
    v1, v2, v3, v4 = int(string_ip_list[0]), int(string_ip_list[1]), int(string_ip_list[2]), int(string_ip_list[3])
    return (v1 << 24) + (v2 << 16) + (v3 << 8) + v4

def visualize_setup_time(boot_dict, node_dict, pp_path, mode):
    '''
    Given a boot dict of {node_id: [(neighbor ip address, setup time)...]}
    We visualize another 2D matrix.
    '''
    pass