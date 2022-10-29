import socket
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline 
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# socket
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
PORT = 8000
HOST = '127.0.0.1'
#serversocket.bind((host, port))

"""
class client(Thread):
    def __init__(self, socket, address):
        Thread.__init__(self)
        self.sock = socket
        self.addr = address
        self.start()

    def run(self):
        while 1:
            print('Client sent:', self.sock.recv(1024).decode())
            self.sock.send(pipeline(self.sock.recv(1024).decode()))
"""
def get_pipeline():
    name = "architext/gptj-162M"
    model = AutoModelForCausalLM.from_pretrained(name, use_auth_token=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(name, use_auth_token=True)
    pipeline = TextGenerationPipeline(tokenizer=tokenizer, model=model, max_length=512, num_return_sequences=10)
    return pipeline

pipeline = get_pipeline()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST,PORT))

    while True:
        s.listen()
        print(f'\nListening for messages from port: {PORT}')
        conn, addr = s.accept()

        # Wait for message
        message = conn.recv(1024)
        print(f'Received request:')
        
        generation = pipeline(message)

        print('Inference completed.', end=' ')
        conn.sendall(generation)