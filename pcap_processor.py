import os
import shutil

OSPF_HELLO, DB_DESCRIPTION, LS_REQUEST, LS_UPDATE, LS_ACKNOWLEDGE = '\"1\"', '\"2\"', '\"3\"', '\"4\"', '\"5\"'
EIGRP_UPDATE, EIGRP_QUERY, EIGRP_REPLY, EIGRP_HELLO = '\"1\"', '\"3\"', '\"4\"', '\"5\"'
EIGRP_ACK = '\"0\"'
NO_ROUTING_GRAPH_CHANGE = 0
ROUTING_GRAPH_CHANGE = 1
'''
Kinda strange that there is no opcode 2 for EIGRP...
Also, SIA_REPLY and SIA_QUERY should take opcode 10 and 11 respectively. 
But I don't think it will happen for smaller networks.
'''

def pcap_processor(path, mode):
    # path = "../../../GNS3/projects/test5/project-files/captures/"
    time_intervals = []
    for subpath in os.listdir(path):
        output_path = '../cached_csvs/' + subpath[:-4] + 'csv'
        if mode == 'ospf':
            cmd = 'tshark -r {} -T fields -e frame.number -e frame.time_relative -e ip.src -e ip.dst -e ospf.packet_length -e ospf -e ospf.msg -E header=y -E separator=, -E quote=d -E occurrence=f > {}'
            final_cmd = cmd.format(path + subpath, output_path)
            os.system(final_cmd)
            # Here the csv is generated and it contains the follow columns
            '''
                index 0: frame.number
                index 1: frame.time_relative
                index 2: ip.src
                index 3: ip.dst
                index 4: ospf.packet_length
                index 5: ospf,
                index 6: ospf.msg
            '''
            earliest_db_desc, latest_ls_acknowledge = None, None
            with open(output_path, 'r') as f:
                for idx, row in enumerate(f):
                    if idx:
                        fields = row.split(',')
                        earliest_db_desc = float(fields[1][1:-2]) if fields[6][:-1] == DB_DESCRIPTION and earliest_db_desc == None else earliest_db_desc
                        latest_ls_acknowledge = float(fields[1][1:-2]) if fields[6][:-1] == LS_ACKNOWLEDGE else latest_ls_acknowledge
            if earliest_db_desc != None and latest_ls_acknowledge != None:
                print(f'Changed detected on link {subpath}')
                time_intervals.append((earliest_db_desc, latest_ls_acknowledge))
            elif earliest_db_desc == None and latest_ls_acknowledge != None:
                print(f'Capturing started too late on link {subpath}')
            elif earliest_db_desc != None and latest_ls_acknowledge == None:
                print(f'Capturing end too early on link {subpath}')
            else:
                print(f'No adjacency change on link {subpath}')
        elif mode == 'eigrp':
            cmd = 'tshark -r {} -T fields -e frame.number -e frame.time_relative -e ip.src -e ip.dst -e frame.len -e eigrp -e eigrp.opcode -e eigrp.ack -E header=y -E separator=, -E quote=d -E occurrence=f > {}'
            final_cmd = cmd.format(path + subpath, output_path)
            os.system(final_cmd)
            # Here the csv is generated and it contains the follow columns
            '''
                index 0: frame.number
                index 1: frame.time_relative
                index 2: ip.src
                index 3: ip.dst
                index 4: frame.len
                index 5: eigrp
                index 6: eigrp.opcode
                index 7: eigrp.ack
            '''
            earliest_query, latest_update = None, None
            with open(output_path, 'r') as f:
                for idx, row in enumerate(f):
                    if idx:
                        fields = row.split(',')
                        earliest_query = float(fields[1][1:-2]) if fields[6] == EIGRP_QUERY and earliest_query == None else earliest_query
                        latest_update = float(fields[1][1:-2]) if fields[6] == EIGRP_UPDATE else latest_update
            if earliest_query != None and latest_update != None:
                print(f'Changed detected on link {subpath}')
                time_intervals.append((earliest_query, latest_update))
            elif earliest_query == None and latest_update != None:
                print(f'Capturing started too late on link {subpath}')
            elif earliest_query != None and latest_update == None:
                print(f'Capturing end too early on link {subpath}')
            else:
                print(f'No adjacency change on link {subpath}')


    for filename in os.listdir('../cached_csvs/'):
        file_path = os.path.join('../cached_csvs/', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    time_delta = list(map(lambda x: x[1] - x[0], time_intervals))
    if not time_delta:
        print(f'No change in routing graph, or discontiguous graphs formed! Retrying...')
        return NO_ROUTING_GRAPH_CHANGE, None
    else:
        print(f'network takes at most {max(time_delta)} seconds to converge')
        return ROUTING_GRAPH_CHANGE, max(time_delta)