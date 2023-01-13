from distutils.command import clean
import json
from re import L, S
from xmlrpc.client import ProtocolError
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

import collections
import os
import shutil

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

class Configurator(object):    
    def __init__(self, file_path, config_parent, args):
        self.file_path = file_path
        self.config_parent = config_parent
        with open(self.file_path, "r") as f:
            self.file_config = json.loads(f.read())
        self.flags = args
        self.ip_address_assignment = {}
        self.netmask = 24
        ###########################
        self.adp_port_profile = collections.defaultdict(set)
        self.forwarding_table = collections.defaultdict(set)
        self.edge_set = set()
        ###########################
        self.node_dict, self.adj = self.parse_links_and_nodes()
        self.subnet_ip_dict = {}
        # This dictionary holds the router leader in a router switch 
        # (key --> switch id, value, (router_id, adapter, port))
        self.router_cluster_leader = {}
        self.router_cluster_lookup = {}
        self.router_only_subnets = set()
        ###########################
        self.max_dynamips_adapters = 3
        self.max_dynamips_ports = 2
        self.destination_subnets = {}
        self.forwarding_rules = collections.defaultdict(list)
        self.configure_ips()
        self.r2r_ports = set()
        self.r2r_ips = set()
        self.r2r_ips_dict = collections.defaultdict(set)
        self.backbone_size = None
        self.backbone_ABRs_ports = set()
        ###########################
        try:
            if os.path.exists(f'{self.config_parent}/project-files'):
                shutil.rmtree(f'{self.config_parent}/project-files')
        except:
            raise Exception("project-files folder exists and it cannot be removed.")
        ###########################
        # For each (router, adaptor, port) combo, the ospf area defaults to 0 (unassigned).
        self.ospf_area = collections.defaultdict(int)
        self.configure_forwarding_metadata()
        if 'ospf' not in self.flags:
            self.configure_forwarding()
        else:
            self.backbone_size = input('\033[93m' + 'ACTION REQUIRED: Please input the size of backbone area, positive integer only!\nIf you wish to include all routers into the backbone area, please type all\nYour current selection is: ' + '\033[0m')
            try:
                self.backbone_size = int(self.backbone_size)
                if self.backbone_size < 2:
                    raise Exception('The backbone area size has to be greater than one!')
            except Exception as e:
                if self.backbone_size != 'all':
                    print('\033[91m' + str(e) + '\033[0m')
                    exit(-1)
            if self.backbone_size != 'all':
                self.assign_ospf_area_wrapper()
        ###########################
        
    def assign_ospf_area_wrapper(self) -> None:
        conns = set()
        for key in self.adj:
            for conn in self.adj[key]:
                source_port, destination, destination_port, source_adapter, destination_adapter = conn
                source, dest = (key, source_adapter, source_port), (destination, destination_adapter, destination_port)
                if self.node_dict[key][1] == 'dynamips' or self.node_dict[destination][1] == 'dynamips':
                    sorted_conn = sorted([source, dest])
                    sorted_conn = (sorted_conn[0][0], sorted_conn[0][1], sorted_conn[0][2], sorted_conn[1][0], sorted_conn[1][1], sorted_conn[1][2])
                    if sorted_conn not in conns:
                        conns.add(sorted_conn)

        nodes = set()
        adj = collections.defaultdict(set)
        for s, _, _, d, _, _ in conns:
            nodes.add(s)
            nodes.add(d)
            adj[s].add(d)
            adj[d].add(s)
        
        self.assign_ospf_area(conns, adj)
        return

    def assign_ospf_area_tree(self, real_conns: set, adj: dict) -> None:
        # This function extracts a tree from the connected graph and assigns the remaining edges to different areas. 
        # There are probably more ways to do this, but it gets a low priority. 
        # Turns out we could assign every interface to area 0.
        '''
        TODO: decide whether we need to remove this function in the future. 
        Since it will NOT support router clusters!!!!!
        Multi-line Commented for higher visibility
        '''
        queue = [list(adj.keys())[0]]
        visited = {list(adj.keys())[0]}
        edges = set()
        while queue:
            front = queue.pop(0)
            for neigh in adj[front]:
                if neigh not in visited:
                    visited.add(neigh)
                    if (neigh, front) not in edges:
                        edges.add((front, neigh))
                    queue.append(neigh)

        port_lookup = {}
        for s, s_adaptor, s_port, d, d_adaptor, d_port in real_conns:
            port_lookup[(s, d)] = (s_adaptor, s_port, d_adaptor, d_port)

        backbone_r2r, non_backbone_r2r = set(), set()
        # Get router to router connections that should be INCLUDED into the backbone area 0. 
        for edge_start, edge_end in edges:
            if (edge_start, edge_end) in port_lookup:
                start_adp, start_port, dest_adp, dest_port = port_lookup[(edge_start, edge_end)]
            else:
                dest_adp, dest_port, start_adp, start_port = port_lookup[(edge_end, edge_start)]
            backbone_r2r.add((edge_start, start_adp, start_port, edge_end, dest_adp, dest_port))

        # Get router to router connections that should be EXCLUDED from the backbone area 0. 
        counter = 1
        for s, s_adaptor, s_port, d, d_adaptor, d_port in real_conns:
            if ((s, s_adaptor, s_port, d, d_adaptor, d_port) not in backbone_r2r and
                (d, d_adaptor, d_port, s, s_adaptor, s_port) not in backbone_r2r and 
                (d, d_adaptor, d_port, s, s_adaptor, s_port) not in non_backbone_r2r):
                non_backbone_r2r.add((s, s_adaptor, s_port, d, d_adaptor, d_port))
                self.ospf_area[(s, s_adaptor, s_port)] = counter
                self.ospf_area[(d, d_adaptor, d_port)] = counter
                counter += 1
        
        return

    def assign_ospf_area(self, real_conns: set, adj: dict) -> None:
        # First assigns the backbone area of user-specified size
        # Then assigns incident areas.
        # To be completed 1/7/2023.
        # Could take me a day or two hahaha.

        '''
        Rough Approach for Backbone area assignment. 
        If there are router clusters:
        1. Group all router clusters into cluster candidates. 
        1.1 If there are router clusters that has less than self.backbone_size routers, expand until the size requirement is satisfied.
        1.2 Else we choose the router cluster with the smallest-sized router cluster as the backbone area
        If there aren't any router clusters, or if all router clusters have more routers than specified.  
        2. Choose any router as the anchor, expand until the size requirement is satisfied.
        '''
        port_lookup = {}
        port_usage = collections.defaultdict(set)
        for s, s_adaptor, s_port, d, d_adaptor, d_port in real_conns:
            port_lookup[(s, d)] = (s_adaptor, s_port, d_adaptor, d_port)
            port_usage[s].add((s_adaptor, s_port))
            port_usage[d].add((d_adaptor, d_port))

        backbone_candidate = None
        current_backbone_routers = set()
        if self.router_cluster_leader:
            size_diff = float('inf')
            for key in self.router_cluster_leader:
                if self.backbone_size >= len(adj[key]) and size_diff > self.backbone_size - len(adj[key]):
                    print(self.backbone_size, len(adj[key]))
                    size_diff = self.backbone_size - len(adj[key]) + 1
                    backbone_candidate = key

            if backbone_candidate != None:
                # add all inner facing interfaces into the backbone area.
                for router_id in adj[backbone_candidate]:
                    if (router_id, key) in port_lookup:
                        router_adaptor, router_port, _, _ = port_lookup[(router_id, key)]
                        self.ospf_area[(router_id, router_adaptor, router_port)] = BACKBONE
                        current_backbone_routers.add(router_id)
                    elif (key, router_id) in port_lookup:
                        _, _, router_adaptor, router_port = port_lookup[(key, router_id)]
                        self.ospf_area[(router_id, router_adaptor, router_port)] = BACKBONE
                        current_backbone_routers.add(router_id)
                backbone_size_deficit = self.backbone_size - len(current_backbone_routers)
                if backbone_size_deficit > 0:
                    # find the router that has the most available and non-area-assigned interfaces, then use that router as an anchor. 
                    max_unused_ports = 0
                    anchor = None
                    for router_id in current_backbone_routers:
                        unused_ports = 0
                        if (router_id, 0, 0) not in self.ospf_area and (router_id, 0, 0) in self.r2r_ports:
                            unused_ports += 1
                        if (router_id, 0, 1) not in self.ospf_area and (router_id, 0, 1) in self.r2r_ports:
                            unused_ports += 1
                        if (router_id, 1, 0) not in self.ospf_area and (router_id, 1, 0) in self.r2r_ports:
                            unused_ports += 1
                        if (router_id, 2, 0) not in self.ospf_area and (router_id, 2, 0) in self.r2r_ports:
                            unused_ports += 1
                        if unused_ports > max_unused_ports:
                            max_unused_ports = unused_ports
                            anchor = router_id

                    if anchor == None:
                        raise Exception('\033[91m' + "1, Topology is either detached, or there are not enough routers." + '\033[0m')
                    else:
                        queue, visited = [anchor], {anchor}
                        while queue:
                            front = queue.pop(0)
                            for neigh in adj[front]:
                                if neigh not in visited and self.node_dict[neigh][1] == 'dynamips':
                                    visited.add(neigh)
                                    if (front, neigh) in port_lookup:
                                        front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
                                        self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
                                        self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
                                    else:
                                        # There should not be any exception thrown here if everything goes as planned.
                                        neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
                                        self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
                                        self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
                                    queue.append(neigh)
                                    current_backbone_routers.add(neigh)
                            if self.backbone_size - len(current_backbone_routers) == 0:
                                break

                        if backbone_size_deficit > 0:
                            raise Exception('\033[91m' + "2, Topology is either detached, or there are not enough routers." + '\033[0m')

        if backbone_candidate == None:
            # Keep trying routers that can be used as an anchor, throw exception if this cannot be achieved. 
            assigned = False
            
            for key in adj:
                if self.node_dict[key][1] == 'dynamips':
                    queue, visited = [key], {key}
                    current_backbone_routers.add(key)
                    while queue:
                        front = queue.pop(0)
                        for neigh in adj[front]:
                            if neigh not in visited and self.node_dict[neigh][1] == 'dynamips':
                                '''
                                If neighbor does not belong to any router cluster, we may add routers normally.
                                '''
                                if neigh not in self.router_cluster_lookup:
                                    visited.add(neigh)
                                    if (front, neigh) in port_lookup:
                                        front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
                                        self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
                                        self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
                                    else:
                                        # There should not be any exception thrown here if everything goes as planned.
                                        neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
                                        self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
                                        self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
                                    queue.append(neigh)
                                    current_backbone_routers.add(neigh)
                                else:
                                    '''
                                    If that is not the case, then we have to add all routers in the router cluster. 
                                    If adding these many routers will exceed the backbone size, we will skip this neighbor all together. 
                                    '''
                                    switch_id = self.router_cluster_lookup[neigh]
                                    if len(current_backbone_routers) + len(adj[switch_id]) > self.backbone_size:
                                        continue
                                    else:
                                        # add incident router into current_backbone_cluster
                                        visited.add(neigh)
                                        if (front, neigh) in port_lookup:
                                            front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
                                            self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
                                            self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
                                        else:
                                            # There should not be any exception thrown here if everything goes as planned.
                                            neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
                                            self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
                                            self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
                                        queue.append(neigh)
                                        current_backbone_routers.add(neigh)
                                        # Assign all router interfaces that connect with the switch area 0
                                        for cluster_router in adj[switch_id]:
                                            if (switch_id, cluster_router) in port_lookup:
                                                _, _,  router_adaptor, router_port = port_lookup[(switch_id, cluster_router)]
                                            else:
                                                router_adaptor, router_port, _, _ = port_lookup[(cluster_router, switch_id)]
                                            self.ospf_area[(cluster_router, router_adaptor, router_port)] = BACKBONE
                                            queue.append(router_id)
                                            visited.add(router_id)
                                            current_backbone_routers.add(neigh)
                            if self.backbone_size - len(current_backbone_routers) == 0:
                                assigned = True
                                break
                        
                        if assigned:
                            break

                    if self.backbone_size - len(current_backbone_routers) == 0:
                        assigned = True
                        break
                    else:
                        current_backbone_routers = set()
                        self.ospf_area = collections.defaultdict(int)
                    if assigned:
                        break

            if not assigned:
                raise Exception('\033[91m' + "There are not enough routers, or there is something wrong with the topology." + '\033[0m')
        
        # Scan the backbone area to find the ABRs. 
        for router_id, _, _ in self.ospf_area:
            if (router_id, 0, 0) in self.r2r_ports and (router_id, 0, 0) not in self.ospf_area:
                self.backbone_ABRs_ports.add((router_id, 0, 0))
            elif (router_id, 0, 1) in self.r2r_ports and (router_id, 0, 1) not in self.ospf_area:
                self.backbone_ABRs_ports.add((router_id, 0, 1))
            elif (router_id, 1, 0) in self.r2r_ports and (router_id, 1, 0) not in self.ospf_area:
                self.backbone_ABRs_ports.add((router_id, 1, 0))
            elif (router_id, 2, 0) in self.r2r_ports and (router_id, 2, 0) not in self.ospf_area:
                self.backbone_ABRs_ports.add((router_id, 2, 0))
    

    
        # At this stage, we assume that the backbone area has already been assigned!!!
        '''
        Rough Approach for Fringe area assignment.
        Scan each ABR in the backbone area. 
        If there are interfaces that have not been assigned to any area
            If the interface of the incident router has not been assigned
                3. Assign the unassigned ABR interface and all interfaces of the incident routers an area id, continuously expand until it can be no longer expanded. 
            If the interface of the incident router has already been assigned.
                4. Assign the unassigned ABR interface with the area id assigned to the interface of the incident router.
        '''
        current_assigned_area = 1
        for ABR_router_id, _, _ in self.backbone_ABRs_ports:
            incident_routers = [device for device in adj[ABR_router_id] if self.node_dict[device][1] == 'dynamips']
            for incident_router in incident_routers:
                if (ABR_router_id, incident_router) in port_lookup:
                    ABR_router_adaptor, ABR_router_port, incident_adaptor, incident_port = port_lookup[(ABR_router_id, incident_router)]
                else:
                    incident_adaptor, incident_port, ABR_router_adaptor, ABR_router_port = port_lookup[(incident_router, ABR_router_id)]
                
                if (ABR_router_id, ABR_router_adaptor, ABR_router_port) not in self.ospf_area:
                    if (incident_router, incident_adaptor, incident_port) not in self.ospf_area:
                        # The interface of the incident router has NOT been assigned.
                        self.ospf_area[(ABR_router_id, ABR_router_adaptor, ABR_router_port)] = current_assigned_area
                        self.ospf_area[(incident_router, incident_adaptor, incident_port)] = current_assigned_area
                        queue, visited = [incident_router], {incident_router}
                        while queue:
                            front = queue.pop(0)
                            for neigh in adj[front]:
                                if neigh not in visited and self.node_dict[neigh][1] == 'dynamips':
                                    if (front, neigh) in port_lookup:
                                        front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
                                    else:
                                        neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
                                    visited.add(neigh)
                                    queue.append(neigh)
                                    # assign interfaces.
                                    if (neigh, 0, 0) in self.r2r_ports and (neigh, 0, 0) not in self.ospf_area:
                                        self.ospf_area[(neigh, 0, 0)] = current_assigned_area
                                    if (neigh, 0, 1) in self.r2r_ports and (neigh, 0, 1) not in self.ospf_area:
                                        self.ospf_area[(neigh, 0, 1)] = current_assigned_area
                                    if (neigh, 1, 0) in self.r2r_ports and (neigh, 1, 0) not in self.ospf_area:
                                        self.ospf_area[(neigh, 1, 0)] = current_assigned_area
                                    if (neigh, 2, 0) in self.r2r_ports and (neigh, 2, 0) not in self.ospf_area:
                                        self.ospf_area[(neigh, 2, 0)] = current_assigned_area

                        current_assigned_area += 1

        # Terminal router interfaces that are unassigned will get the area with the lowest number.
        for key in adj:
            if self.node_dict[key][1] == 'dynamips':
                max_area = -1
                if (key, 0, 0) in self.ospf_area:
                    max_area = max(max_area, self.ospf_area[(key, 0, 0)])
                if (key, 0, 1) in self.ospf_area:
                    max_area = max(max_area, self.ospf_area[(key, 0, 1)])
                if (key, 1, 0) in self.ospf_area:
                    max_area = max(max_area, self.ospf_area[(key, 1, 0)])
                if (key, 2, 0) in self.ospf_area:
                    max_area = max(max_area, self.ospf_area[(key, 2, 0)])
                
                # fill in max area to unassigned interfaces.
                if (0, 0) in port_usage[key] and (key, 0, 0) not in self.ospf_area:
                    self.ospf_area[(key, 0, 0)] = max_area
                if (0, 1) in port_usage[key] and (key, 0, 1) not in self.ospf_area:
                    self.ospf_area[(key, 0, 1)] = max_area
                if (1, 0) in port_usage[key] and (key, 1, 0) not in self.ospf_area:
                    self.ospf_area[(key, 1, 0)] = max_area
                if (2, 0) in port_usage[key] and (key, 2, 0) not in self.ospf_area:
                    self.ospf_area[(key, 2, 0)] = max_area

        # Uncomment if bug detected.
        print('Final')
        for a, b, c in self.ospf_area:
            print(f'id: {a}, adaptor: {b}, port: {c}, area: {self.ospf_area[(a, b, c)]}')
        return 
    def find_new_subnet_range(self):
        # This function finds you a 24-netmask subnet range.
        # returns an integer number that helps you to convert to an ip address.
        values = list(self.ip_address_assignment.values())
        # if there is no subnet available, we use the default 10.0.0.1/24 subnet
        # The binary number down below corresponds to 10.0.0.0
        current_max_subnet_in_integer = 0b00001010000000000000000000000000
        if values:
            for idx in range(len(values)):
                raw_ip_string = None
                if isinstance(values[idx], str):
                    raw_ip_string = values[idx]
                else:
                    raw_ip_string = values[idx][0]
                    
                vl = raw_ip_string.split('.')
                v1, v2, v3, v4 = int(vl[0]), int(vl[1]), int(vl[2]), int(vl[3])
                values[idx] = (v1 << 24) + (v2 << 16) + (v3 << 8)
            current_max_subnet_in_integer = max(values)
            
        new_subnet = ((current_max_subnet_in_integer >> (32 - self.netmask)) + 1) << (32 - self.netmask)
        return new_subnet
    
    def find_available_ip_in_subnet(self, subnet_ip, conn_mode = "r2r"):
        '''
        r2r means router to router, which uses a 24 bit netmask.
        s2r means switch to router, which uses a 25 bit netmask.
        '''
        # This function finds ip for the routers. by default it uses 24 bit netmask.
        # subnet_ip takes in a random ip address from that subnet. 
        values = list(self.ip_address_assignment.values())
        for idx in range(len(values)):
            raw_ip_string = None
            if isinstance(values[idx], str):
                raw_ip_string = values[idx]
            else:
                raw_ip_string = values[idx][0]
                
            vl = raw_ip_string.split('.')
            v1, v2, v3, v4 = int(vl[0]), int(vl[1]), int(vl[2]), int(vl[3])
            # convert the existing ip addresses into 32 bit integers and append them into set.
            values[idx] = (v1 << 24) + (v2 << 16) + (v3 << 8) + v4
        
        values = set(values)
        subnet_ip_base_range = None
        subnet_ip_list = subnet_ip.split('.')
        v1, v2, v3, v4 = int(subnet_ip_list[0]), int(subnet_ip_list[1]), int(subnet_ip_list[2]), int(subnet_ip_list[3])
        subnet_ip_base_range = (v1 << 24) + (v2 << 16) + (v3 << 8) + v4
        if conn_mode == "r2r":
            subnet_ip_base_range = subnet_ip_base_range >> (32 - self.netmask)
            subnet_ip_base_range = subnet_ip_base_range << (32 - self.netmask)
        else:
            subnet_ip_base_range = subnet_ip_base_range >> (32 - self.netmask - 1)
            subnet_ip_base_range = subnet_ip_base_range << (32 - self.netmask - 1)
            return self.convert_int_to_ip_address(subnet_ip_base_range + 1)
            
        # Notice: The number of computers may exceed the amount that the subnet supports. 
        # Probably need to warn the user whenever this happens.
        while 1:
            if subnet_ip_base_range not in values and (subnet_ip_base_range & 0x000000FF) % 128 >= 2 :
                break
            else:
                subnet_ip_base_range += 1
                
        return self.convert_int_to_ip_address(subnet_ip_base_range)
    

    def convert_int_to_ip_address(self, integer_ip):
        range1 = integer_ip >> 24
        range2 = (integer_ip >> 16) & 0b0000000011111111 
        range3 = (integer_ip >> 8) & 0b000000000000000011111111 
        range4 = integer_ip & 0b11111111
        return f'{str(range1)}.{str(range2)}.{str(range3)}.{str(range4)}'
    
    def convert_ip_address_to_int(self, string_ip):
        string_ip_list = string_ip.split('.')
        v1, v2, v3, v4 = int(string_ip_list[0]), int(string_ip_list[1]), int(string_ip_list[2]), int(string_ip_list[3])
        return (v1 << 24) + (v2 << 16) + (v3 << 8) + v4
        
        
    def configure_routers(self):
        '''
        Configure router settings and other settings, mostly used on routers.
        '''
        if not os.path.exists(f'{self.config_parent}/project-files'):
            print(os.mkdir(f'{self.config_parent}/project-files', 0o777))
            
        if not os.path.exists(f'{self.config_parent}/project-files/dynamips'):
            print(os.mkdir(f'{self.config_parent}/project-files/dynamips', 0o777))   
            
        for raw_key in self.ip_address_assignment:
            key = raw_key if isinstance(raw_key, str) else raw_key[0]
            if self.node_dict[key][1] == 'dynamips': 
                if not os.path.exists(f'{self.config_parent}/project-files/dynamips/{key}'):
                    print(os.mkdir(f'{self.config_parent}/project-files/dynamips/{key}', 0o777))
                if not os.path.exists(f'{self.config_parent}/project-files/dynamips/{key}/configs'):
                    print(os.mkdir(f'{self.config_parent}/project-files/dynamips/{key}/configs', 0o777))
                

                # TODO: IMPORTANT!!!!!! Change naming rules if you are directly using an imported gns3 topology file.
                index_number = int((self.node_dict[key][0].split('_'))[1])+10
                if not os.path.exists(f'{self.config_parent}/project-files/dynamips/{key}/configs/i{index_number}_startup-config.cfg'):
                    # This file is only generated once.
                    with open(f'{self.config_parent}/project-files/dynamips/{key}/configs/i{index_number}_startup-config.cfg', 'w') as f:
                        f.write("!\n!\n!\n!\n")
                        f.write("service timestamps debug datetime msec\nservice timestamps log datetime msec\nno service password-encryption\n")
                        f.write("!\n")
                        f.write(f"hostname {self.node_dict[key][0]}\n")
                        f.write("!\n")
                        f.write("ip cef\nno ip domain-lookup\nno ip icmp rate-limit unreachable\nip tcp synwait 5\nno cdp log mismatch duplex\n")
                        f.write("!\n")
                        f.write("line con 0\n")
                        f.write(" exec-timeout 0 0\n logging synchronous\n privilege level 15\n no login\n")
                        f.write("line aux 0\n")
                        f.write(" exec-timeout 0 0\n logging synchronous\n privilege level 15\n no login\n")
                        f.write("!\n!\nend\n")
        
        # generate start-up ip.
        interface_dict = collections.defaultdict(list)
        for key in self.node_dict:
            if self.node_dict[key][1] == 'dynamips':
                # configure forwarding rules for each router.
                

                for adp, port in self.adp_port_profile[key]:
                    if (key, adp, port) not in self.ip_address_assignment:
                        interface_dict[key].append(f'''
                                                    interface FastEthernet{adp}/{port}
                                                     no ip address
                                                     shutdown
                                                     duplex auto
                                                     speed auto
                                                     ''')
                    else:
                        # By default, all dynamips ports have 24 bit netmask.
                        interface_dict[key].append(f'''
                                                    interface FastEthernet{adp}/{port}
                                                     ip address {self.ip_address_assignment[(key, adp, port)]} {'255.255.255.0' if (key, adp, port) in self.r2r_ports else '255.255.255.128'}
                                                     duplex auto
                                                     speed auto
                                                    ''')
                            
        for key in self.node_dict:
            if self.node_dict[key][1] == 'dynamips':
                index_number = int((self.node_dict[key][0].split('_'))[1])+10
                with open(f'{self.config_parent}/project-files/dynamips/{key}/configs/i{index_number}_private-config.cfg', 'w') as f:
                    startup_config =   f'''
                                        !
                                        version 12.4
                                        service timestamps debug datetime msec
                                        service timestamps log datetime msec
                                        no service password-encryption
                                        !
                                        hostname {self.node_dict[key][0]}
                                        !
                                        boot-start-marker
                                        boot-end-marker
                                        !
                                        !
                                        no aaa new-model
                                        memory-size iomem 5
                                        no ip icmp rate-limit unreachable
                                        ip cef
                                        !
                                        !
                                        !
                                        !
                                        no ip domain lookup
                                        !
                                        multilink bundle-name authenticated
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        archive
                                         log config
                                          hidekeys
                                        ! 
                                        !
                                        !
                                        !
                                        ip tcp synwait-time 5
                                        !
                                        !
                                        !
                                        !
                                        '''
                    
                    startup_config += '!'.join(interface_dict[key])
                    startup_config += '!\n'
                    
                    if 'ospf' in self.flags:
                        startup_config += 'router ospf 10\n'
                        startup_config += ' log-adjacency-changes\n'
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 0, 0)])} 0.0.0.127 area {self.ospf_area[(key, 0, 0)]}\n' if (key, 0, 0) in self.ip_address_assignment else ""
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 0, 1)])} 0.0.0.127 area {self.ospf_area[(key, 0, 1)]}\n' if (key, 0, 1) in self.ip_address_assignment else ""
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 1, 0)])} 0.0.0.127 area {self.ospf_area[(key, 1, 0)]}\n' if (key, 1, 0) in self.ip_address_assignment else ""
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 2, 0)])} 0.0.0.127 area {self.ospf_area[(key, 2, 0)]}\n' if (key, 2, 0) in self.ip_address_assignment else ""
                        startup_config += f'ip forward-protocol nd\n'
                        startup_config += '!'
                    else:
                        for forward_info in self.forwarding_rules[key]:
                            outbound_packets_subnet, netmask, router_interface = forward_info
                            startup_config +=     f'''
                                                ip forward-protocol nd
                                                ip route {outbound_packets_subnet} {netmask} {router_interface}
                                                !
                                                '''
                    startup_config +=   '''
                                        !
                                        no ip http server
                                        no ip http secure-server
                                        !
                                        no cdp log mismatch duplex
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        control-plane
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        !
                                        line con 0
                                         exec-timeout 0 0
                                         privilege level 15
                                         logging synchronous
                                        line aux 0
                                         exec-timeout 0 0
                                         privilege level 15
                                         logging synchronous
                                        line vty 0 4
                                         login
                                        !
                                        !
                                        end
                                        '''
                    
                    formatted_config = '\n'.join([line.lstrip(' ') for line in startup_config.split('\n')])
                    f.write(formatted_config)
                    
    def configure_ips(self):
        '''
        Configure vpcs settings.
        '''
        # Dynamically allocate ip address.
        self.dynamic_address_allocation()
            
    def configure_forwarding_metadata(self):
        '''
        Populate r2r_ports, r2r_ips, r2r_ips_dict, and router_cluster_leader that can be populated.    
        '''
        for key in self.adj:
            for conn in self.adj[key]:
                source_port, destination, destination_port, source_adapter, destination_adapter = conn
                if self.node_dict[destination][1] == 'dynamips' and self.node_dict[key][1] == 'dynamips': 
                    self.r2r_ports.add((key, source_adapter, source_port))
                    self.r2r_ports.add((destination, destination_adapter, destination_port))
                    self.r2r_ips.add(self.ip_address_assignment[(key, source_adapter, source_port)])
                    self.r2r_ips.add(self.ip_address_assignment[(destination, destination_adapter, destination_port)])
                    self.r2r_ips_dict[key].add(self.ip_address_assignment[(key, source_adapter, source_port)])
                    self.r2r_ips_dict[destination].add(self.ip_address_assignment[(destination, destination_adapter, destination_port)])
                elif self.node_dict[destination][1] == 'ethernet_switch' and self.node_dict[key][1] == 'dynamips':
                    # TODO: Generate modified r2r connections in router clusters. 
                    # TODO: To be completed tomorrow 1/2/2023
                    if destination in self.router_cluster_leader:
                        cluster_leader = self.router_cluster_leader[destination]
                        # To prevent self connection.
                        if cluster_leader[0] != key:
                            self.r2r_ports.add((key, source_adapter, source_port))
                            self.r2r_ports.add(cluster_leader)
                            self.r2r_ips.add(self.ip_address_assignment[(key, source_adapter, source_port)])
                            self.r2r_ips.add(self.ip_address_assignment[cluster_leader])
                            self.r2r_ips_dict[key].add(self.ip_address_assignment[(key, source_adapter, source_port)])
                            self.r2r_ips_dict[cluster_leader[0]].add(self.ip_address_assignment[cluster_leader])
        return


    def configure_forwarding(self):
        '''
        Configure the forwarding rules for a specific router.
        '''
        raw_edge_set = set()
        for conn in self.adj:
            source = conn
            for dest_info in self.adj[conn]:
                if self.node_dict[conn][1] == 'dynamips':
                    source = (conn, dest_info[3], dest_info[0])
                    for profile in self.adp_port_profile[source[0]]:
                        if dest_info[3] != profile[0] or dest_info[0] != profile[1]:
                            if ((source[0], profile[0], profile[1]), source) not in raw_edge_set:
                                raw_edge_set.add((source, (source[0], profile[0], profile[1])))
                            
                dest = dest_info[1]
                if self.node_dict[dest][1] == 'dynamips':
                    dest = (dest, dest_info[4], dest_info[2])
                    for profile in self.adp_port_profile[dest[0]]:
                        if dest_info[4] != profile[0] or dest_info[2] != profile[1]:
                            if ((dest[0], profile[0], profile[1]), dest) not in raw_edge_set:
                                raw_edge_set.add((dest, (dest[0], profile[0], profile[1])))
                
                if (dest, source) not in raw_edge_set:
                    raw_edge_set.add((source, dest))

        # Diagostic Code Block, to be removed / commented
        token_edge_set = set()
        for i in raw_edge_set:
            first, second = None, None
            if isinstance(i[0], tuple):
                first = self.node_dict[i[0][0]][1]
            else:
                first = self.node_dict[i[0]][1]

            if isinstance(i[1], tuple):
                second = self.node_dict[i[1][0]][1]
            else:
                second = self.node_dict[i[1]][1]
            
            if first == 'ethernet_switch' and i[0] in self.router_cluster_leader:
                # Here, i[1] is a dynamips
                if (self.router_cluster_leader[i[0]], i[1]) not in token_edge_set and self.router_cluster_leader[i[0]][0] != i[1][0]:
                    token_edge_set.add((i[1], self.router_cluster_leader[i[0]]))
            elif second == 'ethernet_switch' and i[1] in self.router_cluster_leader:
                # Here, i[0] is a dynamips
                if (self.router_cluster_leader[i[1]], i[0]) not in token_edge_set and self.router_cluster_leader[i[1]][0] != i[0][0]:
                    token_edge_set.add((i[0], self.router_cluster_leader[i[1]]))
            else:
                token_edge_set.add(i)

        graph = nx.Graph()
        nodes = self.ip_address_assignment.keys()
        for e in token_edge_set:
            graph.add_edge(e[0], e[1])

        for start in nodes:
            if isinstance(start, tuple):
                for end in nodes:
                    if start != end:
                        end_ip = self.ip_address_assignment[end]
                        # We include router interface end ip addresses if no router ips have been assigned.
                        if True:
                            path = nx.dijkstra_path(graph, source=start, target=end)
                            for idx in range(len(path)-1):
                                if path[idx] in self.r2r_ports and path[idx+1] in self.r2r_ports:
                                    next_node_ip = self.ip_address_assignment[path[idx+1]]
                                    if next_node_ip not in self.r2r_ips_dict[start[0]]:
                                        # we check whether the next_node_ip is internal. 
                                        # i.e. belongs to another adp/port combination from the same router.
                                        # We also need to check whether we generated multiple forwarding rules for one single end_ip
                                        self.forwarding_table[start].add((end_ip, '255.255.255.0', next_node_ip))
                                    break
        
        # Merge forwarding table entries.
        fw_rules = {}
        for i in self.forwarding_table:
            if i[0] not in fw_rules:
                fw_rules[i[0]] = set()
            fw_rules[i[0]] = fw_rules[i[0]].union(self.forwarding_table[i])

        # TODO: get rid of duplicates, test this code block.
        for i in fw_rules:
            temp_rules = set()
            visited_end_ip = set()
            for r in fw_rules[i]:
                if clean_string_ip_address(r[0]) not in visited_end_ip:
                    visited_end_ip.add(clean_string_ip_address(r[0]))
                    temp_rules.add(r)
            fw_rules[i] = temp_rules

        for router in fw_rules:
            destination_ips = collections.defaultdict(set)
            for end_ip, _, next_node_ip in fw_rules[router]:
                destination_ips[next_node_ip].add(self.convert_ip_address_to_int(end_ip) & 0xFFFFFF00)
            for forward_ip in destination_ips:
                # Make sure that the router does not intra-router forwarding. (We already have checked that)
                for dest_ip in destination_ips[forward_ip]: 
                    self.forwarding_rules[router].append((self.convert_int_to_ip_address(dest_ip), '255.255.255.0', forward_ip))

    def configure_vpcs(self):
        # print(self.ip_address_assignment)
        # Check whether there is a project-file folder in source parent.
        if not os.path.exists(f'{self.config_parent}/project-files'):
            print(os.mkdir(f'{self.config_parent}/project-files', 0o777))
            
        if not os.path.exists(f'{self.config_parent}/project-files/vpcs'):
            print(os.mkdir(f'{self.config_parent}/project-files/vpcs', 0o777))    
            
        for raw_key in self.ip_address_assignment:
            key = raw_key if isinstance(raw_key, str) else raw_key[0]
            # print('within configure_vpcs ', raw_key, key, self.node_dict[key][1])
            if self.node_dict[key][1] == 'vpcs': 
                if not os.path.exists(f'{self.config_parent}/project-files/vpcs/{key}'):
                    print(os.mkdir(f'{self.config_parent}/project-files/vpcs/{key}', 0o777))
                with open(f'{self.config_parent}/project-files/vpcs/{key}/startup.vpc', 'w') as f:
                    # print(f'configuring {key}')
                    f.write(f'set pcname {self.node_dict[key][0]}\n')
                    local_gateway = self.ip_address_assignment[key]
                    local_gateway_list = local_gateway.split('.')
                    local_gateway_list[-1] = '1' if int(local_gateway_list[-1]) < 128 else '129'
                    local_gateway = '.'.join(local_gateway_list)
                    f.write(f'ip {self.ip_address_assignment[key]} {local_gateway} {self.netmask+1}\n')
            
    def dynamic_address_allocation(self):
        # get the number of routers. 
        switches = [(i, self.node_dict[i][0]) for i in self.node_dict if self.node_dict[i][1] == 'ethernet_switch']
        # TODO: change the assumption, probably does not work in the general case.
        # Current assumption is consecutive switches share the same router. 
        # So we do sorting. :(
        switches = list(map(lambda x: x[0], sorted(switches, key = lambda x: x[1])))
        routers = [i for i in self.node_dict if self.node_dict[i][1] == 'dynamips']
        neighboring_router_dict = collections.defaultdict(str)

        # uses dfs to get all elements in the subnet.
        for s in switches:
            queue = [s]
            visited = set()
            visited.add(s)
            while queue:
                front = queue[0]
                for dest in self.adj[front]:
                    if dest[1] not in visited and self.node_dict[dest[1]][1] != 'ethernet_switch':
                        if self.node_dict[dest[1]][1] != 'dynamips':
                            visited.add(dest[1])
                            queue.append(dest[1])
                        else:
                            router_adapter, router_port = dest[4], dest[2]
                            visited.add((dest[1], router_adapter, router_port))
                            queue.append((dest[1], router_adapter, router_port))
                    #####################################################
                    if self.node_dict[dest[1]][1] == 'dynamips':
                        neighboring_router_dict[s] = dest[1]
                    #####################################################
                queue = queue[1:]
            self.subnet_ip_dict[s] = visited

        # TODO: Get a list of subnets that only consist of routers.
        for key in self.subnet_ip_dict:
            # if we find a VPC in the subnet, we abort the operation and move on to another subnet.
            all_router_flag = True
            leader, largest_priority = None, 0
            for item in self.subnet_ip_dict[key]:
                if isinstance(item, str) and self.node_dict[item][1] == 'vpcs':
                    all_router_flag = False

            if all_router_flag == True:
                self.router_only_subnets.add(key)

        # We start with 10.1.1.2 and we build upwards.
        # base_subnet_range = 0b00001010000000010000000100000010
        ########################################################
        base_subnet_range = 0b00001010000000010000000000000010
        prev_router = ''
        ########################################################
        for key in self.subnet_ip_dict:
            ##########################################################################
            if prev_router != neighboring_router_dict[key]:
                base_subnet_range = (base_subnet_range >> (32 - self.netmask)) + 1
                base_subnet_range = (base_subnet_range << (32 - self.netmask)) + 2
                prev_router = neighboring_router_dict[key]
            else:
                base_subnet_range = (base_subnet_range >> (32 - self.netmask - 1)) + 1
                base_subnet_range = (base_subnet_range << (32 - self.netmask - 1)) + 2
            ##########################################################################
            for item in self.subnet_ip_dict[key]:
                if key == item:
                    continue
                ##########################################################################
                if key not in self.router_only_subnets and isinstance(item, tuple) and self.node_dict[item[0]][1] == 'dynamips':
                    self.ip_address_assignment[item] = self.convert_int_to_ip_address(((base_subnet_range >> (32 - self.netmask)) << (32 - self.netmask)) + 1)
                else:
                    self.ip_address_assignment[item] = self.convert_int_to_ip_address(base_subnet_range)
                ##########################################################################
                base_subnet_range += 1
            
            # shift to another subnet range, do not land at x.x.x.1 or x.x.x.129.
            # these addresses will be used as gateway addresses.
            
        # find switch -- router connection, and assign ips to these interfaces.
        # for key in self.subnet_ip_dict:
        #     for item in self.adj[key]:
        #         if self.node_dict[item[1]][1] == 'dynamips':
        #             router_adapter, router_port = item[4], item[2]
        #             if self.subnet_ip_dict[key]:
        #                 sample_subnet_device = None
        #                 for device in self.subnet_ip_dict[key]:
        #                     if device in self.ip_address_assignment:
        #                         sample_subnet_device = device
        #                         break
        
        #                 if not sample_subnet_device:
        #                     # Breaks here.
        #                     raise NotImplemented
                    
        #                 subnet_ip = self.ip_address_assignment[sample_subnet_device]
        #                 # TODO: convert self.find_available_ip_in_subnet to include netmask as a parameter. 
        #                 # In this case, it should be 25. 
        #                 # Also, create a data structure to count for this. 
        #                 available_ip = self.find_available_ip_in_subnet(subnet_ip, "s2r")
        #                 self.ip_address_assignment[(item[1], router_adapter, router_port)] = available_ip

        # print(self.ip_address_assignment)
                        
        # find router -- router connection. 
        # for each router pair, we use a new subnet with a netmask of 24.
        for r in routers:
            for item in self.adj[r]:
                if self.node_dict[item[1]][1] == 'dynamips':
                    source_router_adapter, source_router_port = item[3], item[0]
                    destination = item[1]
                    dest_router_adapter, dest_router_port = item[4], item[2]
                    if (r, source_router_adapter, source_router_port) not in self.ip_address_assignment and \
                        (destination, dest_router_adapter, dest_router_port) not in self.ip_address_assignment:
                        new_subnet_range = self.find_new_subnet_range()
                        # TODO: convert self.find_available_ip_in_subnet to include netmask as a parameter. 
                        # In this case, it should be 24. 
                        # Also, create a data structure to account for this. 
                        self.ip_address_assignment[(r, source_router_adapter, source_router_port)] = self.convert_int_to_ip_address(new_subnet_range+2)
                        self.ip_address_assignment[(destination, dest_router_adapter, dest_router_port)] = self.convert_int_to_ip_address(new_subnet_range+3)

        # TODO: NEW! Elect a router cluster leader.
        for key in self.subnet_ip_dict:
            # if we find a VPC in the subnet, we abort the operation and move on to another subnet.
            all_router_flag = True
            leader, largest_priority = None, 0
            routers_in_local_router_cluster = set()
            for item in self.subnet_ip_dict[key]:
                if isinstance(item, str) and self.node_dict[item][1] == 'vpcs':
                    all_router_flag = False
                    break
                elif isinstance(item, tuple) and self.node_dict[item[0]][1] == 'dynamips':
                    # Check ip assignment on all router interfaces
                    # TODO: remove hardcoding if it is deemed to be necessary!!!!!
                    largest_local_priority = 0
                    if (item[0], 0, 0) in self.ip_address_assignment:
                        largest_local_priority = max(largest_local_priority, self.convert_ip_address_to_int(self.ip_address_assignment[(item[0], 0, 0)]))
                    if (item[0], 0, 1) in self.ip_address_assignment:
                        largest_local_priority = max(largest_local_priority, self.convert_ip_address_to_int(self.ip_address_assignment[(item[0], 0, 1)]))
                    if (item[0], 1, 0) in self.ip_address_assignment:
                        largest_local_priority = max(largest_local_priority, self.convert_ip_address_to_int(self.ip_address_assignment[(item[0], 1, 0)]))
                    if (item[0], 2, 0) in self.ip_address_assignment:
                        largest_local_priority = max(largest_local_priority, self.convert_ip_address_to_int(self.ip_address_assignment[(item[0], 2, 0)]))
                    if largest_local_priority > largest_priority:
                        leader = item
                        largest_priority = largest_local_priority
                    routers_in_local_router_cluster.add(item)
                    
            if all_router_flag:
                self.router_cluster_leader[key] = leader
                for item in routers_in_local_router_cluster:
                    self.router_cluster_lookup[item] = key

    def parse_links_and_nodes(self):
        links = self.file_config['topology']['links']
        nodes = self.file_config['topology']['nodes']
        node_dict = {}
        adj = collections.defaultdict(set)
        for n in nodes:
            node_dict[n['node_id']] = [n['name'], n['node_type']]

        for l in links:
            source, destination = l['nodes'][0]['node_id'], l['nodes'][1]['node_id']
            source_port, destination_port = l['nodes'][0]['port_number'], l['nodes'][1]['port_number']
            source_adapter, destination_adapter = l['nodes'][0]['adapter_number'], l['nodes'][1]['adapter_number']
            adj[source].add((source_port, destination, destination_port, source_adapter, destination_adapter))
            adj[destination].add((destination_port, source, source_port, destination_adapter, source_adapter))
            self.adp_port_profile[source].add((source_adapter, source_port))
            self.adp_port_profile[destination].add((destination_adapter, destination_port))
        return node_dict, adj
    
    def breadth_first_search(self):
        # return the amount of edges. 
        source = None
        # Find the start point for the bfs traversal.
        for key in self.adj:
            for conn in self.adj[key]:
                source_port, destination, destination_port, source_adapter, destination_adapter = conn
                source, dest = (key, source_adapter, source_port), (destination, destination_adapter, destination_port)
                if self.node_dict[key][1] != 'dynamips':
                    source = key
                break
                    
        queue = [source]
        visited = set()
        visited.add(source)
        
        while queue:
            front = queue[0]
            key, adp, port = None, None, None 
            if isinstance(front, str):
                key = front
            else:
                key, adp, port = front[0], front[1], front[2]
                for val in self.adp_port_profile[key]:
                    if val != (adp, port) and (key, val[0], val[1]) not in visited:
                        queue.append((key, val[0], val[1]))
                        visited.add((key, val[0], val[1]))
                        self.edge_set.add(((key, adp, port), (key, val[0], val[1])))

            for conn in self.adj[key]:
                source_port, destination, destination_port, source_adapter, destination_adapter = conn
                dest = (destination, destination_adapter, destination_port)
                if self.node_dict[destination][1] != 'dynamips':
                    dest = destination
                if dest not in visited:
                    queue.append(dest)
                    visited.add(dest)
                if adp != None and port != None:
                    if source_adapter == adp and source_port == port and (dest, (key, adp, port)) not in self.edge_set:
                        self.edge_set.add(((key, adp, port), dest))
                elif (dest, key) not in self.edge_set:
                    self.edge_set.add((key, dest))
                    
            queue.pop(0)