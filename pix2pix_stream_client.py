import os, sys
import time
import json
import io


import numpy as np
import cv2
from PIL import Image

from cv2_windows import cv2_windows
from websocket_client import ws_client
from webcam_capture import cam_capture




def read_config(config_filename='client_config.json'):
    with open(config_filename, 'r') as f:
        ml = f.readlines()

    ml_uc = [l[:l.find('#')] for l in ml]
    
    l_uc = ''.join( ml_uc )

    l_json = l_uc.replace("\n", ' ').replace("'", '"').replace('(', '[').replace(')', ']').strip()

    
    try:
        cfg_d = json.loads( l_json )
    except json.decoder.JSONDecodeError as e:
        print(' - ERROR, read_config: Error on caracter: "{}"'.format( l_json[e.pos] ) )
        
        global err
        err = e
        raise e
        

    return cfg_d



def img2bytes(img):
    image      = Image.fromarray(img)
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr



def update_video(ws, msg):
##    print('update_video:', type(msg))

    try:
        ws.n_imgs_sent -= 1
        if type(msg) is bytes:
            img = Image.open( io.BytesIO(msg) )
            img_array = np.array(img, dtype=np.uint8)
            ws.window.update_img(img_array)
            
        else:
            print(' - WARNING, {}.'.format(msg), file=sys.stderr)

    except Exception as e:
        print(' - ERROR, update_video: unhandled:', e, file=sys.stderr)
        
    return None





class pix2pix_stream_handler:
    def __init__(self,
                 target_fps=2,
                 host='localhost',
                 port=8000,
                 cam_index=None,
                 win_name='Model Prediction',
                 win_shape=(600,600),
                 cap_shape=(256,256),
                 pts=[[398,441],[202,521],[449,599],[267,698]],
                 password=None):
        
        self.target_fps = target_fps
        self.host = host
        self.port = port
        self.cam_index = cam_index
        self.win_name  = win_name
        self.win_shape = win_shape
        self.cap_shape = cap_shape
        self.pts       = pts
        self.password  = password
        
        self.client = None

        # Object for cam capture
        self.cam = cam_capture(cam_index=self.cam_index,
                               cap_shape=self.cap_shape,
                               pts=self.pts)
        
        self.window = cv2_windows()
        return None


    def connect(self):
        self.client = ws_client(host=self.host,
                                port=self.port,
                                on_message_function=update_video,
                                password=self.password)

        
        
        self.client.start()
        self.client.ws.window = self.window

        self.client.ws.n_imgs_sent = 0

        
        
        self.window.open_window(win_name=self.win_name,
                                win_shape=self.win_shape)
        
        
        return None


    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None

        self.cam.close()

        return None



    def capture_forever(self):

        if self.client.ws is None:
            raise Exception(' - ERROR, capture_forever, the connection is not established.')
        
        while True:
            if self.client.ws is not None:
                img = self.cam.capture()
                
                if self.client.ws.n_imgs_sent < 5:
                    img_bytes = img2bytes(img)
                    self.client.send(img_bytes)
                    self.client.ws.n_imgs_sent += 1
                    
                else:
                    print(' - WARNING, capture_forever: Oversent imgs, you can try lower the target fps ...', file=sys.stderr)
                    
                time.sleep( 1 / self.target_fps )
            else:
                print(' - ERROR, Connection closed, capture_forever will stop!!!', file=sys.stderr)
                break





if __name__ == '__main__':

    client_cfg_d = read_config(config_filename='client_config.json')

    sh = pix2pix_stream_handler( **client_cfg_d )
    sh.connect()
    
    time.sleep(1.0)
##    sh.client.send(img_bytes)
    sh.capture_forever()







    
