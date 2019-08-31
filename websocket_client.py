import websocket
import threading
import time
import json
import os, sys


import numpy as np
import cv2
import io
from PIL import Image

from matplotlib import pyplot as plt

def img2bytes(img):
    image      = Image.fromarray(img)
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr



def on_message(ws, message):
    print('on_message:', message)


def on_error(ws, error):
    print("- Se produjo un ERROR: " + str(error))
    

def on_close(ws):
    print("on_close: ### closed ###")
    self = ws.self
    self.close()
    ws.self.connected = False
    return None
    
    
def on_open(ws):    
    print("- Conectado al Servidor !!!")
    ws.self.connected = True


def start_new_ws(ws):
    ws.run_forever()
    return None




class cv2_window:
    def __init__(self, win_shape=(600,600), win_name='Prediction'):
        self.win_shape = win_shape
        self.win_img = 128*np.ones( self.win_shape, dtype=np.uint8 )
        self.th = None
        self.keeprunning = False
        self.win_name = win_name
        
        return None
    

    def open_window(self):
        if self.th is None:
            self.th = threading.Thread( target=self._win_update )
            self.keeprunning = True
            self.th.start()

        else:
            print(' - WARNING, A window is open now, please close it before start a new one.', file=sys.stderr)

        return None


    def close_windows(self):
        self.keeprunning = False
        return None


    def _win_update(self):

        while self.keeprunning:
            cv2.imshow(self.win_name, self.win_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.keeprunning = False
        cv2.destroyAllWindows()
        self.th = None

        return None


    def update_img(self, img):
        """ the image needs to be in RGB mode """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.win_img = cv2.resize(img, self.win_shape)
        return None






class ws_client:
    def __init__(self,
                 host='localhost',
                 port=8000,
                 use_ssl=False,
                 on_message_function=None):

        self.host    = host
        self.port    = port
        self.use_ssl = use_ssl
        self.ws      = None

        self.connected = False

        if on_message_function is None:
            self.on_message_function = on_message
        else:
            self.on_message_function = on_message_function

        
        return None

        
    def start(self):
        if self.ws is None:
            websocket.enableTrace(False)
            
            self.ws = websocket.WebSocketApp("ws{}://{}:{}".format('s' if self.use_ssl else '', self.host, self.port),
                                             on_message = self.on_message_function,
                                             on_error = on_error,
                                             on_close = on_close)
            self.ws.on_open = on_open
            self.ws.self    = self

            self.th = threading.Thread(target=start_new_ws, args=(self.ws,))
            self.th.start()
        return None
    

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

        return None
    

    def send(self, msg='Hello !!!'):
        if self.ws is not None and self.connected:
            if type(msg) is str:
                self.ws.send(msg, 1)
            elif type(msg) is bytes:
                self.ws.send(msg, 2)
            else:
                print(' - WARNING send: msg type not supported', file=sys.stderr)
                
        else:
            print(' - WARNING send: ws closed, unable to send msg', file=sys.stderr)

        return None
            


def update_video(ws, msg):
##    print('update_video:', type(msg))

    ws.n_imgs_sent -= 1
    if type(msg) is bytes:
        img = Image.open( io.BytesIO(msg) )

        img_array = np.array(img, dtype=np.uint8)

        ws.window.update_img(img_array)
        
    else:
        print(' - WARNING, no binary data in msg to show.', file=sys.stderr)

    return None







class stream_handler:
    def __init__(self,
                 target_fps=2,
                 host='localhost',
                 port=8000,
                 cam_index=None,
                 win_shape=(600,600)):
        
        self.target_fps = target_fps
        self.host = host
        self.port = port
        self.client = None
        
        if cam_index is not None:
            self.cap = cv2.VideoCapture(cam_index)
        else:
            self.cap = None

        self.window = cv2_window(win_shape=win_shape)
        return None


    def connect(self):
        self.client = ws_client(host=self.host,
                                port=self.port,
                                on_message_function=update_video)

        
        
        self.client.start()
        self.client.ws.window = self.window

        self.client.ws.n_imgs_sent = 0
        
        self.window.open_window()
        
        
        return None

    
    def capture_cam(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        else:
            img = np.zeros( (256,256,3), dtype=np.uint8 )
            print(' - WARNING, cam is closed.', file=sys.stderr)

        return img


    def read_file(self, filename='../pix2pix_wrapper/A/input_img.jpg'):
        try:
            image = cv2.imread(filename)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            img = np.zeros( (256,256,3), dtype=np.uint8 )
            print(' - WARNING, unable to read image {}.'.format(filename), file=sys.stderr)

        return img


    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()

        return None



    def capture_forever(self):
        global img_bytes
        
        while True:
            if self.cap is None:
                img = self.read_file()
            else:
                img = self.capture_cam()

            if self.client.ws.n_imgs_sent < 5:
                img_bytes = img2bytes(img)
                self.client.send(img_bytes)
                self.client.ws.n_imgs_sent += 1
            else:
                print('Oversent imgs ...', file=sys.stderr)
                
            time.sleep( 1 / self.target_fps )
        
    
if __name__ == '__main__':
    with open('../pix2pix_wrapper/A/input_img.jpg', 'rb') as f:
        img_bytes = f.read()
    print(' - imagen leida!!')

##    client = ws_client()
##    client.start()
##    time.sleep(1)
##    client.send(img_bytes)


    sh = stream_handler(target_fps=5, cam_index=None)
    sh.connect()
    time.sleep(1.0)
##    sh.client.send(img_bytes)
    sh.capture_forever()
    

