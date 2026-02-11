import socket
import time

PC_PORT = 4098

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Buffer grande PRIMA del bind
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)

    # Ascolta su tutte le interfacce
    sock.bind(("", PC_PORT))

    sock.settimeout(1.0)

    print(f"[RX_TEST] listening UDP *:{PC_PORT}")
    t0 = time.time()
    pkts = 0
    bytes_tot = 0

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            pkts += 1
            bytes_tot += len(data)
            if pkts <= 5:
                print(f"[RX_TEST] pkt#{pkts} len={len(data)} from={addr}")
        except socket.timeout:
            dt = time.time() - t0
            print(f"[RX_TEST] timeout... pkts={pkts} bytes={bytes_tot} in {dt:.1f}s")

if __name__ == "__main__":
    main()
