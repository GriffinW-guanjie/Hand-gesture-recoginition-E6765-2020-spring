import argparse

import cv2
import numpy as np
import zmq
import math
import _thread

from constants import PORT
from utils import string_to_image
from local_controller import detectfingers
from local_controller import senddataThread
#from fingerDetect import fingerdetect

globaldata = []



class StreamViewer:
    def __init__(self, port=PORT):
        """
        Binds the computer to a ip address and starts listening for incoming streams.

        :param port: Port which is used for streaming
        """
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.SUB)
        self.footage_socket.bind('tcp://*:' + port)
        self.footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        self.current_frame = None
        self.keep_running = True


    def receive_stream(self, display=True):
        from firebase import firebase
        """
        Displays displayed stream in a window if no arguments are passed.
        Keeps updating the 'current_frame' attribute with the most recent frame, this can be accessed using 'self.current_frame'
        :param display: boolean, If False no stream output will be displayed.
        :return: None
        """
        cap = cv2.VideoCapture(0)
        state = [0, 0, 0, 0, 0]  # fivefingers, 0 miss, 1 straight, 2 bent
        area = [0, 0, 0, 0, 0]
        length = [0, 0, 0, 0, 0]
        application = firebase.FirebaseApplication('https://iotproj-510ee.firebaseio.com', None)

        self.keep_running = True
        while self.footage_socket and self.keep_running:
            try:
                frame = self.footage_socket.recv_string()
                self.current_frame = string_to_image(frame)
                imgs = []
                blank_image = np.zeros((frame.shape[0], frame.shape[1], 3), np.float32)
                for i in range(0,3):
                    img = string_to_image(self.footage_socket.recv_string())
                    imgs.append(img.astype(np.float32))
                for img in imgs:
                    blank_image = img/3 + blank_image
                aveimg = blank_image.astype(np.uint8)
                img = aveimg
                width = img.shape[1]
                height = img.shape[0]
                thumb, forefinger, midfinger, ringfinger, pinky, rectarea, length_c, tops = detectfingers(img)

                fingers = [thumb, forefinger, midfinger, ringfinger, pinky]

                for i, finger in enumerate(fingers):

                    if finger != 6:
                        # print(length_c[finger], length[i])
                        if state[i] == 0:
                            state[i] = 1
                        elif state[i] == 2 and length_c[finger] >= 1.2 * length[i]:
                            # if length_c[finger] >= 1.2 * length[i] or rectarea[finger] >= 1.2 * area[i]:
                            state[i] = 1
                        elif state[i] == 1 and length_c[finger] <= 0.86 * length[i]:
                            # if length_c[finger] <= 0.8 * length[i] or rectarea[finger] >= 1.2 * area[i]:
                            # print(length_c[finger], length[i])

                            x = int(tops[finger][0] * 408 / width)
                            # print(width, tops[finger][0], x)
                            # print(height)
                            y = int((height - tops[finger][1]) * 408 / height)
                            data = {'x': x, 'y': y}
                            print(data)
                            globaldata.append(data)
                            state[i] = 2
                        length[i] = length_c[finger]


                        area[i] = rectarea[finger]
                    else:
                        state[i] = 0

            except KeyboardInterrupt:
                cv2.destroyAllWindows()
                break
        print("Streaming Stopped!")

    def stop(self):
        """
        Sets 'keep_running' to False to stop the running loop if running.
        :return: None
        """
        self.keep_running = False

def main():
    port = PORT

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port',
                        help='The port which you want the Streaming Viewer to use, default'
                             ' is ' + PORT, required=False)

    args = parser.parse_args()
    if args.port:
        port = args.port

    stream_viewer = StreamViewer(port)
    stream_viewer.receive_stream()


if __name__ == '__main__':
    _thread.start_new_thread(senddataThread, (1, ))
    main()
