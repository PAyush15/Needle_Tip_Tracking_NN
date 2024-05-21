import socket

HOST = '127.0.0.1'
PORT = 65432

def process_values(data):
    values = data.strip().split()
    try:
        # Check if the number of values is correct
        if len(values) != 2:
            raise ValueError("Incorrect number of values")
        result = float(values[0]) + float(values[1])
        return result
    except ValueError as e:
        print("Invalid data format received:", e)
        return None

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind((HOST, PORT))
        s.listen()
        print("Server listening on port", PORT)
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                decoded_data = data.decode()
                if not data:
                    break
                result = process_values(decoded_data.rstrip('\x00'))
                if result is not None:
                    print("Processed result:", result)
                    conn.sendall(f"Processed result: {result}".encode())
                else:
                    # Optionally, handle invalid data
                    pass
    except OSError as e:
        print("Error:", e)
