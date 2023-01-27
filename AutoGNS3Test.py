import gns3fy
import sys
import telnetlib

HOST = "127.0.0.1"

# Define the server object to establish the connection
gns3_server = gns3fy.Gns3Connector("http://127.0.0.1:3080")

# Define the lab you want to load and assign the server connector
lab = gns3fy.Project(name="rctest", connector=gns3_server)

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
lab.open()

# Verify the stats
print(lab.stats)
print("\n")

# key-value format: node_name (e.g. c_8) : {node_id:xxxxx-xxxxx-xxxxx-..., port_num:5xxx, node_ip:10.x.x.x}, 
# can be edited according to demands
node_dict = {}


# Read the names and status of all the nodes in the project
for node in lab.nodes:
    print(node)
    print("\n")
    node_name = node.name
    node_info = {"node_id": node.node_id, "port_num":node.console}
    telnetObj=telnetlib.Telnet(HOST,node.console)
    message = ("show ip\n").encode('ascii')
    telnetObj.write(message)
    output=telnetObj.read_until(match=b"aaaa",timeout=6) # this line now read all output in connect terminal in 6 sec.
    print(output) # "IP/MASK     : 10.1.1.3/25" gives the ip of the node
    node_info["node_ip"] = None # change this value to extracted ip address
    telnetObj.close()

# TODO: using node dict to ping other nodes, using belowing commands:
telnetObj=telnetlib.Telnet(HOST,node.console)
ip_to_ping = None # modify here to input node ip that is going to be pinged
message = ("ping " + ip_to_ping + "\n" ).encode('ascii')
telnetObj.write(message)
output=telnetObj.read_until(match=b"aaaa",timeout=6) # same as above, need to find a better way to extract info
print(output) # seems like: b'ping 10.1.4.2\r\n84 bytes from 10.1.4.2 icmp_seq=1 ttl=62 time=35.172 ms\r\n84 bytes from 10.1.4.2 icmp_seq=2 ttl=62 time=81.043 ms\r\n
              # 84 bytes from 10.1.4.2 icmp_seq=3 ttl=62 time=81.239 ms\r\n84 bytes from 10.1.4.2 icmp_seq=4 ttl=62 time=81.358 ms\r\n84 bytes from 10.1.4.2 icmp_seq=5 ttl=62 time=80.673 ms\r\n\r\n\rc_10> '

# close the project
lab.close()
