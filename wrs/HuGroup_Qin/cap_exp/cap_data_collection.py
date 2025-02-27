import time
import re
from wrs.HuGroup_Qin.driver.robot_driver.dobot_api import DobotApiDashboard, DobotApiMove, DobotApi, MyType, alarmAlarmJsonFile
from time import sleep
import keyboard  # This module is used to detect key presses
import matplotlib.pyplot as plt
import numpy as np
import threading
import csv

# 全局变量(当前坐标)
current_actual = [-1]
algorithm_queue = -1
enableStatus_robot = -1
robotErrorState = False
globalLockValue = threading.Lock()
def connect_robot(ip = "192.168.5.100"):
    try:
        ip = ip
        dashboard_p = 29999
        move_p = 30003
        feed_p = 30004
        print("正在建立连接...")
        dashboard = DobotApiDashboard(ip, dashboard_p)
        move = DobotApiMove(ip, move_p)
        feed = DobotApi(ip, feed_p)
        print(">.<连接成功>!<")
        return dashboard, move, feed
    except Exception as e:
        print(":(连接失败:(")
        raise e

def ClearRobotError(dashboard: DobotApiDashboard):
    global robotErrorState
    dataController, dataServo = alarmAlarmJsonFile()    # 读取控制器和伺服告警码
    while True:
        globalLockValue.acquire()
        if robotErrorState:
            numbers = re.findall(r'-?\d+', dashboard.GetErrorID())
            numbers = [int(num) for num in numbers]
            if (numbers[0] == 0):
                if (len(numbers) > 1):
                    for i in numbers[1:]:
                        alarmState = False
                        if i == -2:
                            print("机器告警 机器碰撞 ", i)
                            alarmState = True
                        if alarmState:
                            continue
                        for item in dataController:
                            if i == item["id"]:
                                print("机器告警 Controller errorid", i,
                                      item["zh_CN"]["description"])
                                alarmState = True
                                break
                        if alarmState:
                            continue
                        for item in dataServo:
                            if i == item["id"]:
                                print("机器告警 Servo errorid", i,
                                      item["zh_CN"]["description"])
                                break

                    choose = input("输入1, 将清除错误, 机器继续运行: ")
                    if int(choose) == 1:
                        dashboard.ClearError()
                        sleep(0.01)
                        dashboard.Continue()

        else:
            if int(enableStatus_robot) == 1 and int(algorithm_queue) == 0:
                dashboard.Continue()
        globalLockValue.release()
        sleep(5)

def GetFeed(feed: DobotApi):
    global current_actual
    global algorithm_queue
    global enableStatus_robot
    global robotErrorState
    hasRead = 0
    while True:
        data = bytes()
        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp
        hasRead = 0
        feedInfo = np.frombuffer(data, dtype=MyType)
        if hex((feedInfo['test_value'][0])) == '0x123456789abcdef':
            globalLockValue.acquire()
            # Refresh Properties
            current_actual = feedInfo["tool_vector_actual"][0]
            algorithm_queue = feedInfo['run_queued_cmd'][0]
            enableStatus_robot = feedInfo['enable_status'][0]
            robotErrorState = feedInfo['error_status'][0]
            globalLockValue.release()
        sleep(0.001)

def parse_data(data_str):
    """ 解析单个数据字符串，提取六个力和扭矩分量。"""
    match = re.search(r'\{([^}]*)\}', data_str)
    if match:
        data_list = match.group(1).split(',')
        return list(map(float, data_list[:6]))
    else:
        return None

def compute_zero_offsets(data_samples):
    """ 计算给定数据样本的平均值作为零点校准偏移。"""
    sum_data = [0.0] * 6
    for data in data_samples:
        parsed_data = parse_data(data)
        if parsed_data:
            sum_data = [sum_data[i] + parsed_data[i] for i in range(6)]

    count = len(data_samples)
    return [x / count for x in sum_data]

def apply_zero_offset(real_data_str, offsets):
    """ 使用计算出的偏移校正实际数据。"""
    real_data = parse_data(real_data_str)
    if real_data:
        corrected_data = [real_data[i] - offsets[i]for i in range(6)]
        return corrected_data
    return None

def save_data_to_txt(data, filename="corrected_data.txt"):
    """ 保存数据到.txt文件。"""
    with open(filename, "w") as file:
        for data_entry in data:
            file.write(f"{data_entry}\n")
    print(f"数据已保存到 {filename}")


def JointMovJ(move: DobotApiMove, joint_list: list):
    move.JointMovJ(joint_list[0], joint_list[1], joint_list[2],
                joint_list[3], joint_list[4], joint_list[5])


if __name__ == '__main__':

    dashboard, move, feed = connect_robot(ip = "192.168.5.100")
    feed_thread = threading.Thread(target=GetFeed, args=(feed,))
    feed_thread.daemon = True
    feed_thread.start()
    feed_thread1 = threading.Thread(target=ClearRobotError, args=(dashboard,))
    feed_thread1.daemon = True
    feed_thread1.start()
    print("begin enable process...")
    dashboard.EnableRobot()
    print("complete enable process")

    example_num = 30
    exam_num = 0

    rotation_num = 40
    rot_num = 0
    rotation_started = False

    collect_data = []
    global recording
    global recorded_data
    recording = False
    recorded_data = []

    dashboard.SpeedFactor(5) # 准静态假设
    current_jnv = np.array(parse_data(dashboard.GetAngle()))
    target_jnv6 = -180
    target_jnv = current_jnv + np.array([0, 0, 0, 0, 0, target_jnv6])

    filename = "Cap6-3.txt"
    test_flag = "test"

    if test_flag == "test":
        while True:
            force_data = dashboard.GetSixForceData()
            if exam_num < example_num:
                collect_data.append(force_data)
                exam_num += 1
            else:
                offsets = compute_zero_offsets(collect_data)
                print(f"Offsets: {offsets}")
                break

        while True:
            # Check for spacebar press
            if keyboard.is_pressed("space"):
                recording = not recording  # Toggle recording state
                if recording:
                    print("\n >>>>>>>>>>>开始记录数据<<<<<<<<<<<< \n.")
                    recorded_data = []  # Clear the previously recorded data
                else:
                    print("\n >>>>>>>>>>>停止记录数据<<<<<<<<<<<<<. \n")
                    # Save recorded data when stopped
                    save_data_to_txt(recorded_data, filename)

                    break
                # To avoid detecting multiple presses from one spacebar press
                sleep(0.2)

            force_data = dashboard.GetSixForceData()
            corrected_data = apply_zero_offset(force_data, offsets)
            print(f"Recorded data: {corrected_data}")

            if recording and corrected_data:
                # record force data and joint angle
                if rot_num > rotation_num and not rotation_started:
                    rotation_started = True  # 设置标志位防止重复执行
                    print(" %%%%% 开始旋转 %%%%% ")
                    JointMovJ(move, target_jnv)
                if not rot_num > rotation_num:
                    rotation_started = False  # 按键松开后重置标志位
                    rot_num += 1
                recorded_data.append([corrected_data, parse_data(dashboard.GetAngle())])

        print("experiment is over!")

    elif test_flag == "recover":
        dashboard.SpeedFactor(50)  # 准静态假设
        current_jnv = np.array(parse_data(dashboard.GetAngle()))
        target_jnv6 = 180
        target_jnv = current_jnv + np.array([0, 0, 0, 0, 0, target_jnv6])
        JointMovJ(move, target_jnv)