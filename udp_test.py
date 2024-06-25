import socket
import time
import numpy as np
import cv2

MAX_UDP_PACKET_SIZE = 500

# udp通信の設定
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
address = ('192.168.0.255', 12345)
# address = ('localhost', 12345)
        
def send_udp(image):
    """
    画像をUDPで送信する関数
    """
    byte = cv2.imencode('.png', image)[1].tobytes()
    print(len(byte))
    chunks = split_data(byte, MAX_UDP_PACKET_SIZE)
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        header = i.to_bytes(4, 'big') + total_chunks.to_bytes(4, 'big')
        is_last_chunk = (1 if i == total_chunks - 1 else 0).to_bytes(1, 'big')
        udp_packet = header + is_last_chunk + chunk
        udp_socket.sendto(udp_packet, address)

def split_data(data, size):
    """データを指定されたサイズに分割する"""
    return [data[i:i+size] for i in range(0, len(data), size)]

img = np.zeros((256, 256, 3), dtype=np.uint8)
count = 0
while True:
    img[:, :, :] = count
    count = (count + 1) % 256
    send_udp(img)
    time.sleep(0.1)