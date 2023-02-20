from gc import collect
from re import L, S
from threading import currentThread
from turtle import distance
from lib import *
import os
import shutil

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
        self.virtual_link_setup = collections.defaultdict(set)
        self.graph = None
        if 'metadata' in self.flags:
            return
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
        if 'ospf' not in self.flags and 'eigrp' not in self.flags:
            self.configure_forwarding()
        elif 'eigrp' not in self.flags:
            self.backbone_size = input('\033[93m' + 'ACTION REQUIRED: Please input the max size for each area, positive integer greater than 1 only!\nIf you wish to include all routers into the backbone area, please type all\nYour current selection is: ' + '\033[0m')
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
        
        self.assign_ospf_area_balanced(conns, adj)
        return

    '''
    Deprecated: Please remove in future iterations!!!!
    '''
    # def assign_ospf_area_tree(self, real_conns: set, adj: dict) -> None:
    #     # This function extracts a tree from the connected graph and assigns the remaining edges to different areas. 
    #     # There are probably more ways to do this, but it gets a low priority. 
    #     # Turns out we could assign every interface to area 0.
    #     '''
    #     TODO: decide whether we need to remove this function in the future. 
    #     Since it will NOT support router clusters!!!!!
    #     Multi-line Commented for higher visibility
    #     '''
    #     queue = [list(adj.keys())[0]]
    #     visited = {list(adj.keys())[0]}
    #     edges = set()
    #     while queue:
    #         front = queue.pop(0)
    #         for neigh in adj[front]:
    #             if neigh not in visited:
    #                 visited.add(neigh)
    #                 if (neigh, front) not in edges:
    #                     edges.add((front, neigh))
    #                 queue.append(neigh)

    #     port_lookup = {}
    #     for s, s_adaptor, s_port, d, d_adaptor, d_port in real_conns:
    #         port_lookup[(s, d)] = (s_adaptor, s_port, d_adaptor, d_port)

    #     backbone_r2r, non_backbone_r2r = set(), set()
    #     # Get router to router connections that should be INCLUDED into the backbone area 0. 
    #     for edge_start, edge_end in edges:
    #         if (edge_start, edge_end) in port_lookup:
    #             start_adp, start_port, dest_adp, dest_port = port_lookup[(edge_start, edge_end)]
    #         else:
    #             dest_adp, dest_port, start_adp, start_port = port_lookup[(edge_end, edge_start)]
    #         backbone_r2r.add((edge_start, start_adp, start_port, edge_end, dest_adp, dest_port))

    #     # Get router to router connections that should be EXCLUDED from the backbone area 0. 
    #     counter = 1
    #     for s, s_adaptor, s_port, d, d_adaptor, d_port in real_conns:
    #         if ((s, s_adaptor, s_port, d, d_adaptor, d_port) not in backbone_r2r and
    #             (d, d_adaptor, d_port, s, s_adaptor, s_port) not in backbone_r2r and 
    #             (d, d_adaptor, d_port, s, s_adaptor, s_port) not in non_backbone_r2r):
    #             non_backbone_r2r.add((s, s_adaptor, s_port, d, d_adaptor, d_port))
    #             self.ospf_area[(s, s_adaptor, s_port)] = counter
    #             self.ospf_area[(d, d_adaptor, d_port)] = counter
    #             counter += 1
        
    #     return

    '''
    Older version of ospf area assigner. 
    '''
    # def assign_ospf_area(self, real_conns: set, adj: dict) -> None:
    #     # First assigns the backbone area of user-specified size
    #     # Then assigns incident areas.

    #     '''
    #     Rough Approach for Backbone area assignment. 
    #     If there are router clusters:
    #     1. Group all router clusters into cluster candidates. 
    #     1.1 If there are router clusters that has less than self.backbone_size routers, expand until the size requirement is satisfied.
    #     1.2 Else we choose the router cluster with the smallest-sized router cluster as the backbone area
    #     If there aren't any router clusters, or if all router clusters have more routers than specified.  
    #     2. Choose any router as the anchor, expand until the size requirement is satisfied.
    #     '''
    #     port_lookup = {}
    #     port_usage = collections.defaultdict(set)
    #     for s, s_adaptor, s_port, d, d_adaptor, d_port in real_conns:
    #         port_lookup[(s, d)] = (s_adaptor, s_port, d_adaptor, d_port)
    #         port_usage[s].add((s_adaptor, s_port))
    #         port_usage[d].add((d_adaptor, d_port))

    #     backbone_candidate = None
    #     current_backbone_routers = set()
    #     if self.router_cluster_leader:
    #         size_diff = float('inf')
    #         for key in self.router_cluster_leader:
    #             if self.backbone_size >= len(adj[key]) and size_diff > self.backbone_size - len(adj[key]):
    #                 print(self.backbone_size, len(adj[key]))
    #                 size_diff = self.backbone_size - len(adj[key]) + 1
    #                 backbone_candidate = key

    #         if backbone_candidate != None:
    #             # add all inner facing interfaces into the backbone area.
    #             for router_id in adj[backbone_candidate]:
    #                 if (router_id, key) in port_lookup:
    #                     router_adaptor, router_port, _, _ = port_lookup[(router_id, key)]
    #                     self.ospf_area[(router_id, router_adaptor, router_port)] = BACKBONE
    #                     current_backbone_routers.add(router_id)
    #                 elif (key, router_id) in port_lookup:
    #                     _, _, router_adaptor, router_port = port_lookup[(key, router_id)]
    #                     self.ospf_area[(router_id, router_adaptor, router_port)] = BACKBONE
    #                     current_backbone_routers.add(router_id)
    #             backbone_size_deficit = self.backbone_size - len(current_backbone_routers)
    #             if backbone_size_deficit > 0:
    #                 # find the router that has the most available and non-area-assigned interfaces, then use that router as an anchor. 
    #                 max_unused_ports = 0
    #                 anchor = None
    #                 for router_id in current_backbone_routers:
    #                     unused_ports = 0
    #                     if (router_id, 0, 0) not in self.ospf_area and (router_id, 0, 0) in self.r2r_ports:
    #                         unused_ports += 1
    #                     if (router_id, 0, 1) not in self.ospf_area and (router_id, 0, 1) in self.r2r_ports:
    #                         unused_ports += 1
    #                     if (router_id, 1, 0) not in self.ospf_area and (router_id, 1, 0) in self.r2r_ports:
    #                         unused_ports += 1
    #                     if (router_id, 2, 0) not in self.ospf_area and (router_id, 2, 0) in self.r2r_ports:
    #                         unused_ports += 1
    #                     if unused_ports > max_unused_ports:
    #                         max_unused_ports = unused_ports
    #                         anchor = router_id

    #                 if anchor == None:
    #                     raise Exception('\033[91m' + "1, Topology is either detached, or there are not enough routers." + '\033[0m')
    #                 else:
    #                     queue, visited = [anchor], {anchor}
    #                     while queue:
    #                         front = queue.pop(0)
    #                         for neigh in adj[front]:
    #                             if neigh not in visited and self.node_dict[neigh][1] == 'dynamips':
    #                                 visited.add(neigh)
    #                                 if (front, neigh) in port_lookup:
    #                                     front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
    #                                     self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
    #                                     self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
    #                                 else:
    #                                     # There should not be any exception thrown here if everything goes as planned.
    #                                     neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
    #                                     self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
    #                                     self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
    #                                 queue.append(neigh)
    #                                 current_backbone_routers.add(neigh)
    #                         if self.backbone_size - len(current_backbone_routers) == 0:
    #                             break

    #                     if backbone_size_deficit > 0:
    #                         raise Exception('\033[91m' + "2, Topology is either detached, or there are not enough routers." + '\033[0m')

    #     if backbone_candidate == None:
    #         # Keep trying routers that can be used as an anchor, throw exception if this cannot be achieved. 
    #         assigned = False
            
    #         for key in adj:
    #             if self.node_dict[key][1] == 'dynamips':
    #                 queue, visited = [key], {key}
    #                 current_backbone_routers.add(key)
    #                 while queue:
    #                     front = queue.pop(0)
    #                     for neigh in adj[front]:
    #                         if neigh not in visited and self.node_dict[neigh][1] == 'dynamips':
    #                             '''
    #                             If neighbor does not belong to any router cluster, we may add routers normally.
    #                             '''
    #                             if neigh[0] not in self.router_cluster_lookup:
    #                                 visited.add(neigh)
    #                                 if (front, neigh) in port_lookup:
    #                                     front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
    #                                     self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
    #                                     self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
    #                                 else:
    #                                     # There should not be any exception thrown here if everything goes as planned.
    #                                     neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
    #                                     self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
    #                                     self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
    #                                 queue.append(neigh)
    #                                 current_backbone_routers.add(neigh)
    #                             else:
    #                                 '''
    #                                 If that is not the case, then we have to add all routers in the router cluster. 
    #                                 If adding these many routers will exceed the backbone size, we will skip this neighbor all together. 
    #                                 '''
    #                                 switch_id = self.router_cluster_lookup[neigh[0]]
    #                                 if len(current_backbone_routers) + len(adj[switch_id]) > self.backbone_size:
    #                                     continue
    #                                 else:
    #                                     # add incident router into current_backbone_cluster
    #                                     visited.add(neigh)
    #                                     if (front, neigh) in port_lookup:
    #                                         front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
    #                                         self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
    #                                         self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
    #                                     else:
    #                                         # There should not be any exception thrown here if everything goes as planned.
    #                                         neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
    #                                         self.ospf_area[(front, front_adaptor, front_port)] = BACKBONE
    #                                         self.ospf_area[(neigh, neigh_adaptor, neigh_port)] = BACKBONE
    #                                     queue.append(neigh)
    #                                     current_backbone_routers.add(neigh)
    #                                     # Assign all router interfaces that connect with the switch area 0
    #                                     for cluster_router in adj[switch_id]:
    #                                         if (switch_id, cluster_router) in port_lookup:
    #                                             _, _,  router_adaptor, router_port = port_lookup[(switch_id, cluster_router)]
    #                                         else:
    #                                             router_adaptor, router_port, _, _ = port_lookup[(cluster_router, switch_id)]
    #                                         self.ospf_area[(cluster_router, router_adaptor, router_port)] = BACKBONE
    #                                         queue.append(router_id)
    #                                         visited.add(router_id)
    #                                         current_backbone_routers.add(neigh)
    #                         if self.backbone_size - len(current_backbone_routers) == 0:
    #                             assigned = True
    #                             break
                        
    #                     if assigned:
    #                         break

    #                 if self.backbone_size - len(current_backbone_routers) == 0:
    #                     assigned = True
    #                     break
    #                 else:
    #                     current_backbone_routers = set()
    #                     self.ospf_area = collections.defaultdict(int)
    #                 if assigned:
    #                     break

    #         if not assigned:
    #             raise Exception('\033[91m' + "There are not enough routers, or there is something wrong with the topology." + '\033[0m')
        
    #     # Scan the backbone area to find the ABRs. 
    #     for router_id, _, _ in self.ospf_area:
    #         if (router_id, 0, 0) in self.r2r_ports and (router_id, 0, 0) not in self.ospf_area:
    #             self.backbone_ABRs_ports.add((router_id, 0, 0))
    #         elif (router_id, 0, 1) in self.r2r_ports and (router_id, 0, 1) not in self.ospf_area:
    #             self.backbone_ABRs_ports.add((router_id, 0, 1))
    #         elif (router_id, 1, 0) in self.r2r_ports and (router_id, 1, 0) not in self.ospf_area:
    #             self.backbone_ABRs_ports.add((router_id, 1, 0))
    #         elif (router_id, 2, 0) in self.r2r_ports and (router_id, 2, 0) not in self.ospf_area:
    #             self.backbone_ABRs_ports.add((router_id, 2, 0))
    
    #     # At this stage, we assume that the backbone area has already been assigned!!!
    #     '''
    #     Rough Approach for Fringe area assignment.
    #     Scan each ABR in the backbone area. 
    #     If there are interfaces that have not been assigned to any area
    #         If the interface of the incident router has not been assigned
    #             3. Assign the unassigned ABR interface and all interfaces of the incident routers an area id, continuously expand until it can be no longer expanded. 
    #         If the interface of the incident router has already been assigned.
    #             4. Assign the unassigned ABR interface with the area id assigned to the interface of the incident router.
    #     '''
    #     current_assigned_area = 1
    #     for ABR_router_id, _, _ in self.backbone_ABRs_ports:
    #         incident_routers = [device for device in adj[ABR_router_id] if self.node_dict[device][1] == 'dynamips']
    #         for incident_router in incident_routers:
    #             if (ABR_router_id, incident_router) in port_lookup:
    #                 ABR_router_adaptor, ABR_router_port, incident_adaptor, incident_port = port_lookup[(ABR_router_id, incident_router)]
    #             else:
    #                 incident_adaptor, incident_port, ABR_router_adaptor, ABR_router_port = port_lookup[(incident_router, ABR_router_id)]
                
    #             if (ABR_router_id, ABR_router_adaptor, ABR_router_port) not in self.ospf_area:
    #                 if (incident_router, incident_adaptor, incident_port) not in self.ospf_area:
    #                     # The interface of the incident router has NOT been assigned.
    #                     self.ospf_area[(ABR_router_id, ABR_router_adaptor, ABR_router_port)] = current_assigned_area
    #                     self.ospf_area[(incident_router, incident_adaptor, incident_port)] = current_assigned_area
    #                     queue, visited = [incident_router], {incident_router}
    #                     while queue:
    #                         front = queue.pop(0)
    #                         for neigh in adj[front]:
    #                             if neigh not in visited and self.node_dict[neigh][1] == 'dynamips':
    #                                 if (front, neigh) in port_lookup:
    #                                     front_adaptor, front_port, neigh_adaptor, neigh_port = port_lookup[(front, neigh)]
    #                                 else:
    #                                     neigh_adaptor, neigh_port, front_adaptor, front_port = port_lookup[(neigh, front)]
    #                                 visited.add(neigh)
    #                                 queue.append(neigh)
    #                                 # assign interfaces.
    #                                 if (neigh, 0, 0) in self.r2r_ports and (neigh, 0, 0) not in self.ospf_area:
    #                                     self.ospf_area[(neigh, 0, 0)] = current_assigned_area
    #                                 if (neigh, 0, 1) in self.r2r_ports and (neigh, 0, 1) not in self.ospf_area:
    #                                     self.ospf_area[(neigh, 0, 1)] = current_assigned_area
    #                                 if (neigh, 1, 0) in self.r2r_ports and (neigh, 1, 0) not in self.ospf_area:
    #                                     self.ospf_area[(neigh, 1, 0)] = current_assigned_area
    #                                 if (neigh, 2, 0) in self.r2r_ports and (neigh, 2, 0) not in self.ospf_area:
    #                                     self.ospf_area[(neigh, 2, 0)] = current_assigned_area

    #                     current_assigned_area += 1

    #     # Terminal router interfaces that are unassigned will get the area with the lowest number.
    #     for key in adj:
    #         if self.node_dict[key][1] == 'dynamips':
    #             max_area = -1
    #             if (key, 0, 0) in self.ospf_area:
    #                 max_area = max(max_area, self.ospf_area[(key, 0, 0)])
    #             if (key, 0, 1) in self.ospf_area:
    #                 max_area = max(max_area, self.ospf_area[(key, 0, 1)])
    #             if (key, 1, 0) in self.ospf_area:
    #                 max_area = max(max_area, self.ospf_area[(key, 1, 0)])
    #             if (key, 2, 0) in self.ospf_area:
    #                 max_area = max(max_area, self.ospf_area[(key, 2, 0)])
                
    #             # fill in max area to unassigned interfaces.
    #             if (0, 0) in port_usage[key] and (key, 0, 0) not in self.ospf_area:
    #                 self.ospf_area[(key, 0, 0)] = max_area
    #             if (0, 1) in port_usage[key] and (key, 0, 1) not in self.ospf_area:
    #                 self.ospf_area[(key, 0, 1)] = max_area
    #             if (1, 0) in port_usage[key] and (key, 1, 0) not in self.ospf_area:
    #                 self.ospf_area[(key, 1, 0)] = max_area
    #             if (2, 0) in port_usage[key] and (key, 2, 0) not in self.ospf_area:
    #                 self.ospf_area[(key, 2, 0)] = max_area

    #     ospf_area_key_schedule = sorted(list(self.ospf_area.keys()))
    #     for a, b, c in ospf_area_key_schedule:
    #         print(f'{a}, {b}, {c} assigned to area: {self.ospf_area[(a, b, c)]}')
    #     return 

    def assign_ospf_area_balanced(self, real_conns: set, adj: dict) -> None:
        # G shows the router connections ONLY.
        G = generate_router_graph(real_conns, self.node_dict)
        if not G.edges():
            # If there is no router connections AT ALL. Then, we have to group every router cluster to area 0.
            return
        # split routers into areas.
        _, inv_area_assignment, incident_set = graph_partition_wrapper(G, self.backbone_size)
        currently_assigned_area = max(list(inv_area_assignment.values())) + 1
        border_routers = set()
        area_router_count = collections.defaultdict(int)
        router_cluster_area = {}
        for router in inv_area_assignment:
            if router in self.router_cluster_lookup:
                router_cluster_area[self.router_cluster_lookup[router]] = inv_area_assignment[router]
            
        for n1, n2 in incident_set:
            border_routers.add(n1)
            border_routers.add(n2)

        port_lookup = {}
        port_usage = collections.defaultdict(set)
        for s, s_adaptor, s_port, d, d_adaptor, d_port in real_conns:
            port_lookup[(s, d)] = (s_adaptor, s_port, d_adaptor, d_port)
            port_usage[s].add((s_adaptor, s_port))
            port_usage[d].add((d_adaptor, d_port))
        
        # If the connection is not with a router cluster, assign area accordingly.
        for router in inv_area_assignment:
            if router not in border_routers:
                area_router_count[inv_area_assignment[router]] += 1
                if (router, 0, 0) in self.r2r_ports:
                    self.ospf_area[(router, 0, 0)] = inv_area_assignment[router]
                elif (router, 0, 1) in self.r2r_ports:
                    self.ospf_area[(router, 0, 1)] = inv_area_assignment[router]
                elif (router, 1, 0) in self.r2r_ports:
                    self.ospf_area[(router, 1, 0)] = inv_area_assignment[router]
                elif (router, 2, 0) in self.r2r_ports:
                    self.ospf_area[(router, 2, 0)] = inv_area_assignment[router]

        for r1, r2 in incident_set:
            if (r1, r2) in port_lookup:
                r1_adp, r1_port, r2_adp, r2_port = port_lookup[(r1, r2)]
            else:
                r2_adp, r2_port, r1_adp, r1_port = port_lookup[(r2, r1)]

            if r1 in self.router_cluster_lookup or r2 in self.router_cluster_lookup:
                continue
            # Balance the router area assignment for border routers.
            if area_router_count[inv_area_assignment[r1]] >= area_router_count[inv_area_assignment[r2]]:
                self.ospf_area[(r1, r1_adp, r1_port)] = inv_area_assignment[r2]
                self.ospf_area[(r2, r2_adp, r2_port)] = inv_area_assignment[r2]
                area_router_count[inv_area_assignment[r2]] += 2
            else:        
                self.ospf_area[(r1, r1_adp, r1_port)] = inv_area_assignment[r1]
                self.ospf_area[(r2, r2_adp, r2_port)] = inv_area_assignment[r1]
                area_router_count[inv_area_assignment[r1]] += 2
            
        # Fill in ports for router clusters.
        for r1, r2 in port_lookup:
            r1_adp, r1_port, r2_adp, r2_port = port_lookup[(r1, r2)]
            if r1 in self.router_cluster_lookup and r2 in self.router_cluster_lookup:
                s1, s2 = self.router_cluster_lookup[r1], self.router_cluster_lookup[r2]
                self.ospf_area[(r1, r1_adp, r1_port)] = router_cluster_area[s1]
                self.ospf_area[(r2, r2_adp, r2_port)] = router_cluster_area[s1]
            
            elif r1 in self.router_cluster_lookup:
                s1 = self.router_cluster_lookup[r1]
                self.ospf_area[(r1, r1_adp, r1_port)] = router_cluster_area[s1]
                if self.node_dict[r2][1] == 'dynamips':
                    self.ospf_area[(r2, r2_adp, r2_port)] = router_cluster_area[s1]

            elif r2 in self.router_cluster_lookup:
                s2 = self.router_cluster_lookup[r2]
                if self.node_dict[r1][1] == 'dynamips':
                    self.ospf_area[(r1, r1_adp, r1_port)] = router_cluster_area[s2]
                self.ospf_area[(r2, r2_adp, r2_port)] = router_cluster_area[s2]

        # Fill in remaining ports.
        for key in self.node_dict:            
            if self.node_dict[key][1] == 'dynamips':
                if (0, 0) in port_usage[key] and (key, 0, 0) not in self.ospf_area:
                    if key in self.router_cluster_lookup:
                        self.ospf_area[(key, 0, 0)] = router_cluster_area[self.router_cluster_lookup[key]]
                    else:
                        self.ospf_area[(key, 0, 0)] = inv_area_assignment[key]
                                    
                if (0, 1) in port_usage[key] and (key, 0, 1) not in self.ospf_area:
                    if key in self.router_cluster_lookup:
                        self.ospf_area[(key, 0, 1)] = router_cluster_area[self.router_cluster_lookup[key]]
                    else:
                        self.ospf_area[(key, 0, 1)] = inv_area_assignment[key]
                if (1, 0) in port_usage[key] and (key, 1, 0) not in self.ospf_area:
                    if key in self.router_cluster_lookup:
                        self.ospf_area[(key, 1, 0)] = router_cluster_area[self.router_cluster_lookup[key]]
                    else:
                        self.ospf_area[(key, 1, 0)] = inv_area_assignment[key]
                if (2, 0) in port_usage[key] and (key, 2, 0) not in self.ospf_area:
                    if key in self.router_cluster_lookup:
                        self.ospf_area[(key, 2, 0)] = router_cluster_area[self.router_cluster_lookup[key]]
                    else:
                        self.ospf_area[(key, 2, 0)] = inv_area_assignment[key]

        '''
        Algorithm:
        1. Find the Area border routers of backbone. 
        2. Perform BFS, set up virtual link along the way. 
        '''
        area_routers_dict = collections.defaultdict(set)
        for a, b, c in self.ospf_area:
            area_routers_dict[self.ospf_area[(a, b, c)]].add(a)

        peripheral_areas = set()
        border_router_ports = collections.defaultdict(set)
        dist_to_backbone = {BACKBONE: 0}
        areas = set(self.ospf_area.values())
        area_adjacencies = collections.defaultdict(set)
        interfaces_between_areas = collections.defaultdict(set)
        for a, b, c in self.ospf_area:
            for area in areas:
                if a in area_routers_dict[area] and self.ospf_area[(a, b, c)] != area:
                    if area == 0:
                        peripheral_areas.add(self.ospf_area[(a, b, c)])
                        dist_to_backbone[self.ospf_area[(a, b, c)]] = 1
                    # Establish adjacency.
                    border_router_ports[area].add((a, b, c))
                    area_adjacencies[area].add(self.ospf_area[(a, b, c)])
                    s_area, d_area = sorted([self.ospf_area[(a, b, c)], area])
                    interfaces_between_areas[(s_area, d_area)].add((a, b, c))

                    
        '''
        We start assigning virtual-links from the peripheral areas.
        '''
        # for stem_area in peripheral_areas:
        #     queue = [stem_area]
        #     while queue:
        #         transit_area = queue.pop(0)
        #         for router_id, router_adaptor, router_port in border_router_ports[transit_area]:
        #             border_area = self.ospf_area[(router_id, router_adaptor, router_port)]
        #             if border_area not in peripheral_areas and border_area not in dist_to_backbone:
        #                 # Connect the base area to the border area with a virtual-link.
        #                 minimum_distance, destination_router_id = float('inf'), None
        #                 for virtual_router_id, virtual_router_adaptor, virtual_router_port in border_router_ports[transit_area]:
        #                     if (router_id, router_adaptor, router_port) != (virtual_router_id, virtual_router_adaptor, virtual_router_port):
        #                         connection_area = self.ospf_area[(virtual_router_id, virtual_router_adaptor, virtual_router_port)]
        #                         if connection_area in dist_to_backbone:
        #                             if dist_to_backbone[connection_area] < minimum_distance:
        #                                 minimum_distance = dist_to_backbone[connection_area]
        #                                 destination_router_id = virtual_router_id
        #                 # At this stage, we should've determined the destination router, now what we can do here is stitch the routers together.
        #                 # For each corresponding router, we store a set of tuples (transit_area: int, )
        #                 self.virtual_link_setup[destination_router_id].add((transit_area, self.get_router_ospf_id(router_id)))
        #                 self.virtual_link_setup[router_id].add((transit_area, self.get_router_ospf_id(destination_router_id)))

        #                 # Update border area's distance to the backbone area.
        #                 # Add border area to queue. 
        #                 dist_to_backbone[border_area] = minimum_distance + 1
        #                 queue.append(border_area)

        ag = nx.Graph()
        for s in area_adjacencies:
            for d in area_adjacencies[s]:
                ag.add_edge(s, d)

        bypassed_area_triplets = set()
        for area in areas:
            best_path = nx.dijkstra_path(ag, source=area, target=0)
            for index in range(len(best_path)-2):
                source, transit, end = best_path[index], best_path[index+1], best_path[index+2]
                if (source, transit, end) not in bypassed_area_triplets:
                    bypassed_area_triplets.add((source, transit, end))
                    shortest_path_distance, shortest_path = float('inf'), None
                    # Find the shortest path between all routers in set (source, transit) and (transit, end). 
                    # Then, we establish the shortest virtual-link between source and end.
                    source_side_start, source_side_end = sorted([source, transit])
                    dest_side_start, dest_side_end = sorted([transit, end])
                    for source_router_id, sa, sp in interfaces_between_areas[(source_side_start, source_side_end)]:
                        for dest_router_id, da, dp in interfaces_between_areas[(dest_side_start, dest_side_end)]:
                            path_length = nx.dijkstra_path_length(self.graph, (source_router_id, sa, sp), (dest_router_id, da, dp))
                            if path_length < shortest_path_distance:
                                shortest_path_distance = path_length
                                shortest_path = [source_router_id, dest_router_id]
                    self.virtual_link_setup[shortest_path[1]].add((transit, self.get_router_ospf_id(shortest_path[0])))
                    self.virtual_link_setup[shortest_path[0]].add((transit, self.get_router_ospf_id(shortest_path[1])))

        ospf_area_key_schedule = sorted(list(self.ospf_area.keys()))
        for a, b, c in ospf_area_key_schedule:
            print(f'{a}, {b}, {c} assigned to area: {self.ospf_area[(a, b, c)]}')
        print()
        ospf_border_router_schedule = sorted(list(self.virtual_link_setup.keys()))
        for router_id in ospf_border_router_schedule:
            for transit_area, dest_ospf_router_id in self.virtual_link_setup[router_id]:
                print(f'router: {router_id}, transit_area: {transit_area}, destination ospf router id: {dest_ospf_router_id}')
        return

    def get_router_ospf_id(self, router_id) -> str:
        max_ip_address = float('-inf')
        if (router_id, 0, 0) in self.ip_address_assignment:
            max_ip_address = max(max_ip_address, convert_ip_address_to_int(self.ip_address_assignment[(router_id, 0, 0)]))
        if (router_id, 0, 1) in self.ip_address_assignment:
            max_ip_address = max(max_ip_address, convert_ip_address_to_int(self.ip_address_assignment[(router_id, 0, 1)]))
        if (router_id, 1, 0) in self.ip_address_assignment:
            max_ip_address = max(max_ip_address, convert_ip_address_to_int(self.ip_address_assignment[(router_id, 1, 0)]))
        if (router_id, 2, 0) in self.ip_address_assignment:
            max_ip_address = max(max_ip_address, convert_ip_address_to_int(self.ip_address_assignment[(router_id, 2, 0)]))
        
        ret_val = convert_int_to_ip_address(max_ip_address)
        return ret_val

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
            return convert_int_to_ip_address(subnet_ip_base_range + 1)
            
        # Notice: The number of computers may exceed the amount that the subnet supports. 
        # Probably need to warn the user whenever this happens.
        while 1:
            if subnet_ip_base_range not in values and (subnet_ip_base_range & 0x000000FF) % 128 >= 2 :
                break
            else:
                subnet_ip_base_range += 1
                
        return convert_int_to_ip_address(subnet_ip_base_range)
            
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
                        for transit_area, destination_ip in self.virtual_link_setup[key]:
                            startup_config += f' area {transit_area} virtual-link {destination_ip}\n'
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 0, 0)])} 0.0.0.127 area {self.ospf_area[(key, 0, 0)]}\n' if (key, 0, 0) in self.ip_address_assignment else ""
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 0, 1)])} 0.0.0.127 area {self.ospf_area[(key, 0, 1)]}\n' if (key, 0, 1) in self.ip_address_assignment else ""
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 1, 0)])} 0.0.0.127 area {self.ospf_area[(key, 1, 0)]}\n' if (key, 1, 0) in self.ip_address_assignment else ""
                        startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 2, 0)])} 0.0.0.127 area {self.ospf_area[(key, 2, 0)]}\n' if (key, 2, 0) in self.ip_address_assignment else ""
                        startup_config += f'ip forward-protocol nd\n'
                        startup_config += '!'
                    elif 'eigrp' in self.flags:
                        startup_config += 'router eigrp 101\n'
                        # find subnets connect to this router and add to network
                        if self.ip_address_assignment.get((key, 0, 0)) != None:
                            startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 0, 0)])} 0.0.0.127\n'
                        if self.ip_address_assignment.get((key, 0, 1)) != None:
                            startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 0, 1)])} 0.0.0.127\n'
                        if self.ip_address_assignment.get((key, 1, 0)) != None:
                            startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 1, 0)])} 0.0.0.127\n'
                        if self.ip_address_assignment.get((key, 2, 0)) != None:
                            startup_config += f' network {clean_string_ip_address(self.ip_address_assignment[(key, 2, 0)])} 0.0.0.127\n'
                        startup_config += 'no auto-summary\n'
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
                    print(self.router_cluster_leader[i[0]])
            elif second == 'ethernet_switch' and i[1] in self.router_cluster_leader:
                # Here, i[0] is a dynamips
                if (self.router_cluster_leader[i[1]], i[0]) not in token_edge_set and self.router_cluster_leader[i[1]][0] != i[0][0]:
                    token_edge_set.add((i[0], self.router_cluster_leader[i[1]]))
                    print(self.router_cluster_leader[i[1]])
            else:
                token_edge_set.add(i)

        self.graph = nx.Graph()
        for e in token_edge_set:
            self.graph.add_edge(e[0], e[1])
        return


    def configure_forwarding(self):
        '''
        Configure the forwarding rules for a specific router.
        '''
        nodes = self.ip_address_assignment.keys()

        for start in nodes:
            if isinstance(start, tuple):
                for end in nodes:
                    if start != end:
                        end_ip = self.ip_address_assignment[end]
                        # We include router interface end ip addresses if no router ips have been assigned.
                        if True:
                            path = nx.dijkstra_path(self.graph, source=start, target=end)
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
                destination_ips[next_node_ip].add(convert_ip_address_to_int(end_ip) & 0xFFFFFF00)
            for forward_ip in destination_ips:
                # Make sure that the router does not intra-router forwarding. (We already have checked that)
                for dest_ip in destination_ips[forward_ip]: 
                    self.forwarding_rules[router].append((convert_int_to_ip_address(dest_ip), '255.255.255.0', forward_ip))

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
                    self.ip_address_assignment[item] = convert_int_to_ip_address(((base_subnet_range >> (32 - self.netmask)) << (32 - self.netmask)) + 1)
                else:
                    self.ip_address_assignment[item] = convert_int_to_ip_address(base_subnet_range)
                ##########################################################################
                base_subnet_range += 1
            
            # shift to another subnet range, do not land at x.x.x.1 or x.x.x.129.
            # these addresses will be used as gateway addresses.
                        
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
                        self.ip_address_assignment[(r, source_router_adapter, source_router_port)] = convert_int_to_ip_address(new_subnet_range+2)
                        self.ip_address_assignment[(destination, dest_router_adapter, dest_router_port)] = convert_int_to_ip_address(new_subnet_range+3)

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
                        largest_local_priority = max(largest_local_priority, convert_ip_address_to_int(self.ip_address_assignment[(item[0], 0, 0)]))
                    if (item[0], 0, 1) in self.ip_address_assignment:
                        largest_local_priority = max(largest_local_priority, convert_ip_address_to_int(self.ip_address_assignment[(item[0], 0, 1)]))
                    if (item[0], 1, 0) in self.ip_address_assignment:
                        largest_local_priority = max(largest_local_priority, convert_ip_address_to_int(self.ip_address_assignment[(item[0], 1, 0)]))
                    if (item[0], 2, 0) in self.ip_address_assignment:
                        largest_local_priority = max(largest_local_priority, convert_ip_address_to_int(self.ip_address_assignment[(item[0], 2, 0)]))
                    if largest_local_priority > largest_priority:
                        leader = item
                        largest_priority = largest_local_priority
                    # routers_in_local_router_cluster.add(item)
                    routers_in_local_router_cluster.add(item[0])
                    
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