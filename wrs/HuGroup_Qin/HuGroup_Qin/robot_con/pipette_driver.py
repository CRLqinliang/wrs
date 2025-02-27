""" 

Author: Liang Qin (qinl.drlrobot@gmail.com)
Created: 20240705 Osaka Univ.

"""
import serial
import time

class pipette:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port, baudrate, timeout=1)
        if self.ser.is_open:
            print(f"Success Connected to Pipette {port} at {baudrate} baudrate")

    def send_data(self, data):
        try:
            # 发送数据
            self.ser.write(data.encode())  # 将字符串编码为字节
            print(f"Data sent: {data}")

        except serial.SerialException as e:
            print(f"Error: {e}")

    def hold(self):
        self.send_data("ma")

    def abosrb(self, sec):
        self.send_data("op")
        time.sleep(sec)
        self.hold()

    def release(self, sec):
        self.send_data("re")
        time.sleep(sec)

    def port_close(self):
        self.ser.close()

    def __del__(self):
        self.ser.close()


if __name__ == "__main__":

    port = 'COM10'
    baudrate = 115200
    pipette = pipette(port, baudrate)

