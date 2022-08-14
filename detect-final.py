import cv2
import time
import sys
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import datetime
import threading
from telegram_utils import send_telegram
from io import StringIO
from pathlib import Path
import streamlit as st
import time
import os
import argparse
from PIL import Image
from st_on_hover_tabs import on_hover_tabs
from streamlit_option_menu import option_menu
import ast
# from streamlit_toggle import st_toggleswitch

st.set_page_config(layout="wide")

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
LAST_ALERT = None
ALERT_TELEGRAM_EACH = 15
MODEL_PATH = "model/person.onnx"
CAM1_PATH = "input/cam1_Trim_2s.mp4"
CAM2_PATH = "input/cam2.mp4"
RESULT_CAM1 = "runs/detect/Output.mp4"
RESULT_CAM2 = "runs/detect/Output1.mp4"
PATH_COORD_CAM1 = 'coord_polygon/point_cam1.txt'
PATH_COORD_CAM2 = 'coord_polygon/point_cam2.txt'
POINTS1 = []
POINTS2 = []

def read_polygon_coord(path_txt, coord_list):
    with open(path_txt) as f:
        lines = f.read().splitlines()
    for line in lines:
        # convert ['a, b'] to [a, b]
        res = ast.literal_eval(line)
        res = np.array(res).tolist()
        coord_list.append(res)
    return coord_list

POINTS1 = read_polygon_coord(PATH_COORD_CAM1, POINTS1)
POINTS2 = read_polygon_coord(PATH_COORD_CAM2, POINTS2)
# print(POINTS1)
# print(POINTS2)
# POINTS2 = [[271, 193], [590, 183], [1065, 585], [903, 686], [474, 689], [271, 193]]


def handle_left_click(event, x, y, flags, points):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])

def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle( frame, (point[0], point[1]), 5, (0,0,255), -1)

    frame = cv2.polylines(frame, [np.int32(points)], False, (255,0, 0), thickness=1)
    return frame

def isInside(points, centroid):
    polygon = Polygon(points)
    centroid = Point(centroid)
    return polygon.contains(centroid)

def alert(img):
    global LAST_ALERT
    # BGR
    cv2.putText(img, "WARNING !!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # New thread to send telegram after 15 seconds
    if (LAST_ALERT is None) or (
            (datetime.datetime.utcnow() - LAST_ALERT).total_seconds() > ALERT_TELEGRAM_EACH):
        LAST_ALERT = datetime.datetime.utcnow()
        cv2.imwrite("alert.png", cv2.resize(img, dsize=(1280,720), fx=0.2, fy=0.2))
        thread = threading.Thread(target=send_telegram)
        thread.start()
    # image with text Warning
    return img

def build_model(is_cuda):
    net = cv2.dnn.readNet(MODEL_PATH)
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detection(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

def load_capture(path_cam):
    capture = cv2.VideoCapture(path_cam)
    return capture

def load_classes():
    class_list = []
    with open("model/classperson.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > SCORE_THRESHOLD):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 


                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45) 

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

total_frames = 0
size = (INPUT_WIDTH, INPUT_HEIGHT)

# Bất kỳ ai đang tìm kiếm cách thuận tiện và mạnh mẽ nhất để ghi tệp MP4 bằng OpenCV hoặc FFmpeg, 
# đều có thể thấy API WriteGear của thư viện Python xử lý video VidGear tiên tiến nhất của tôi hoạt động 
# với cả phần phụ trợ OpenCV và phần phụ trợ FFmpeg và thậm chí hỗ trợ bộ mã hóa GPU . 
# Dưới đây là một ví dụ để mã hóa bằng bộ mã hóa H264 trong WriteGear với phần phụ trợ FFmpeg:

# save with 30 fps
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("output.avi",fourcc, 30.0, (640,640))
from vidgear.gears import WriteGear
output_params = {"-vcodec":"libx264", "-crf": 0, "-preset": "fast"}

writer1 = WriteGear(output_filename = RESULT_CAM1, logging = True, **output_params)
writer2 = WriteGear(output_filename = RESULT_CAM2, logging = True, **output_params)


def intrusion_detection(path_cam):
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
    is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
    net = build_model(is_cuda)
    capture = load_capture(path_cam)

    start = time.time_ns()
    frame_count = 0
    global total_frames
    fps = -1
    
    if path_cam == CAM1_PATH:
        points = POINTS1
    else:
        points = POINTS2
    # points = POINTS
    detect = False
    while True:

        ret, frame = capture.read()
        if frame is None:
            print("End of stream")
            break
        if ret ==True:
            frame = draw_polygon(frame, points)

            if detect:
                inputImage = format_yolov5(frame)
                outs = detection(inputImage, net)

                class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

                frame_count -=- 1
                total_frames = total_frames +1

                for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                    color = colors[int(classid) % len(colors)]
                    cv2.rectangle(frame, box, color, 2)
                    # Rectangle on label person
                    cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)

                    # Create centroid
                    x = round(box[0])
                    y = round(box[1])
                    w = round(box[0]+ box[2])
                    h = round(box[3]+ box[1])

                    centroid = ((x + w) // 2, (y + h) // 2)
                    cv2.circle(frame, centroid, 5, (color), -1)

                    cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))


                    if isInside(points, centroid):
                        alert(frame)

                    
                if frame_count >= 30:
                    end = time.time_ns()
                    fps = 1000000000 * frame_count / (end - start)
                    frame_count = 0
                    start = time.time_ns()
                
                if fps > 0:
                    fps_label = "FPS: %.2f" % fps
                    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            detect = True
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
                
            cv2.imshow("Intrusion Warning", frame)
            # cv2.setMouseCallback('Intrusion Warning', handle_left_click, points)
            if path_cam == CAM1_PATH:
                writer1.write(frame)
            else:
                writer2.write(frame)
        else:
            break
    
def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


def interface():

    # st.header("INTRUSION DETECTION SYSTEM")
    # def header(url):
    #     st.markdown(f'<p style="background-color:#00004d;text-align:center;color:#ff0066;font-size:40px;text-align=center;border-radius:10%;font-family: "Times New Roman", Times, serif;">{url}</marquee></p>', unsafe_allow_html=True)
    # col1, col2, col3 = st.columns(3)
    new_title = '<p style="font-family:sans-serif; background-color: #8B9DA7; color: black; font-size: 42px;"> <b>&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Intrusion Detection System (IDS)🚀</b></p>'
    # with col2:
    st.markdown(new_title, unsafe_allow_html=True)
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSEbOu7Anr-YyDQkbUceLhHj08qi7m0w1nSGQ&usqp=CAU', width=80, caption='August, 2022')
    st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


    with st.sidebar:
        tabs = on_hover_tabs(tabName=['About app', 'Dashboard', 'Exit'], 
                             iconName=['house', 'dashboard', 'logout'],
                             styles = {'navtab': {'background-color':'#8B9DA7',
                                                  'color': 'black',
                                                  
                                                  'font-size': '18px',
                                                  'font-weight': 'bold',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'uppercase'},
                                       'tabOptionsStyle': {':hover :hover': {'color': 'white',
                                                                      'cursor': 'pointer'}},
                                       'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                       'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '50px',
                                                     'padding-left': '30px'}},
                             key="1")

    if tabs =='About app':
        st.balloons()
        st.write("## About app")
        col1, col2 = st.columns(2)
        with col2:
            st.image('https://i.ytimg.com/vi/qdOF5nsqWqA/maxresdefault.jpg')
        with col1:
            st.write('1. **An intrusion detection system** (IDS) is a system that monitors an area for suspicious activity and issues an alert when it is detected.')
            st.write('2. Alert system through sending messages via **Telegram user**.')
            st.write('3. Live tracking of different camera sources to perform intrusion detection using **Artificial intelligence** technology.')
            st.write('4. Follow my github: [link](https://github.com/phuctrang/intrusion_detection)')
    elif tabs == 'Dashboard':
        st.title("Dashboard")

        selected = option_menu(None, ["CAM 1", "CAM 2"], 
        icons=['camera', 'camera'], 
        menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
        "container": {"padding": "0!important", "background-color": "#6498b1"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "pink"},
        "nav-link-selected": {"background-color": "green"},
        })

        if selected == "CAM 1":
            press_button = st.checkbox("View details!")
            if press_button :

                st.write('**+ Type**: Camera-ip-wifi-ezviz-c6n-1080p-2mp-1c2wfr')
                st.write('**+ Location**: 39/54 Ngo May, Quy Nhơn city, Binh Dinh province, Viet Nam')
                
            video_file = open(CAM1_PATH, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            if st.button('Start'):
                intrusion_detection(CAM1_PATH)
            
            if st.button('Show result'):
                video_file1 = open(RESULT_CAM1, 'rb')
                video_bytes1 = video_file1.read()
                st.video(video_bytes1)

        else:
            # st.write('CAM 2')
            press_button1 = st.checkbox("View details")
            if press_button1 :
                st.write('**+ Type**: Camera-ip-wifi-ezviz-c6n-1080p-2mp-1c2wfr')
                st.write('**+ Location**: Quy Nhon University, Binh Dinh province, Viet Nam')
            video_file = open(CAM2_PATH, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            if st.button('Start'):
                intrusion_detection(CAM2_PATH)
            
            if st.button('Show result'):
                video_file2 = open(RESULT_CAM2, 'rb')
                video_bytes2 = video_file2.read()
                st.video(video_bytes2)
    
    elif tabs == 'Exit':
        st.stop()
# print("Total frames: " + str(total_frames))
if __name__ == "__main__":
    interface()
# intrusion_detection(CAM2_PATH)