import socket
PORT = 8765
def main ():
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as client:
        client.setsockopt (socket.SOL_SOCKET,socket.so_REUSEADDR, 1)
        CLIENT.BIND(('',port))
        print(f"listening for broadcast on port{port}..")
        while True
            data, addr = client.recvfrom(1024)
            print (f"[{addr[0]}] {data.decode()}")
if __name__ == "__main__":
    main()