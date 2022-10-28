import time
import socket


# SEND REQUEST TO UPDATE
# Set context
PORT = 8000
HOST = '127.0.0.1'

#context = zmq.Context()
#socket = context.socket(zmq.REQ)
#
# Set timeouts (otherwise GH will freeze when connection fails)
timeout = 1000


# Connect to server
print("Connecting to server...")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST,PORT))
    
    s.sendall(b'run model!')

    #send message to server
    print(f'Sending request...')

    data = s.recv(1024).decode()
    print (data)
server_msg_ = data
# RETURN PROCESSED FILE PATH
filepath_ = r'F:\PhD_Research\Output\CaseStudies\MAP-Elites\pv_urban\qd_setup\minimal-ml-serverCPU\\' + '\output.png'