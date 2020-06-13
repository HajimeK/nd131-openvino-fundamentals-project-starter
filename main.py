"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import nvidia.inference as nv_infer
from nvidia.utils import dboxes300_coco, Encoder
from kuangliu.encoder import DataEncoder
import numpy as np

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def preprocess(n, c, h, w, img):
    input_shape = (n, c, h, w)
    img = cv2.resize(img, (h, w))
    img_data = np.array(img).astype(np.float16)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    norm_img_data = np.zeros(img_data.shape).astype('float16')
    return img_data

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """

    cur_request_id = 0
    current_count = 0
    last_count = 0
    total_count = 0
    start_time = 0

    ### TODO: Load the model through `infer_network` ###
    # Initialise the class
    infer_network = Network()
    # Load the network to IE plugin to get shape of input layer
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          cur_request_id, args.cpu_extension)[1]

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    # Checks for video file
        # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")

    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        # Start asynchronous inference for specified request.
        inf_start = time.time()

        ### TODO: Pre-process the image as needed ###
        img_preprocessed = preprocess(n, c, h, w, frame)

        ### TODO: Start asynchronous inference for specified request ###
        infer_network.exec_net(cur_request_id, img_preprocessed)

        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            ### TODO: Get the results of the inference request ###
            output = infer_network.get_output(cur_request_id, 'DetectionOutput')
            detections = output[0, 0, :, :]

            ### TODO: Extract any desired stats from the results ###
            for detection in detections:
                # If only the cifidence rate is above 0.5, then proceed
                confidence = detection[2]
                if confidence > .5:
                    current_count += 1
                    # detection class
                    idx = detection[1]
                    class_name = coco_classes[idx]
                    log.info(" "+str(idx) + " " + str(confidence) + " " + class_name)
                    if int(idx) == 1: #only person 
                        # Get the box to be displayed
                        axis = detection[3:7] * (initial_w, initial_h, initial_w, initial_h)
                        (start_X, start_Y, end_X, end_Y) = axis.astype(np.int)[:4]
                        cv2.rectangle(frame, (start_X, start_Y), (end_X, end_Y), (0, 55, 255), thickness=2)
                        cv2.putText(frame, class_name, (start_X, start_Y), cv2.FONT_ITALIC, (.0005*initial_w), (0, 0, 255))

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###


            #frame, current_count = ssd_out(frame, result)
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

            # When new person enters the video
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            # Person duration in the video is calculated
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

        current_count = 0

    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
