import numpy as np
import time
import numpy as np
import cv2
from cv2_windows import cv2_windows
import os, sys

class cam_capture():
    def __init__(self,
                 cam_index=0,
                 cap_shape=(256,256),
                 img_path='./test_img.jpeg',
                 pts=[[398,441],[202,521],[449,599],[267,698]] ):
        
        """ If cam_index will read the file instead of the webcam."""

        
        self.cam_index = cam_index
        self.cap_shape = tuple( cap_shape )
        self.img_path  = img_path

        if pts is not None:
            self.pts1 = np.array(pts, dtype=np.float32)
            self.pts2 = np.array([
                                  [0,0],
                                  [0,cap_shape[1]],
                                  [cap_shape[0],0],
                                  [cap_shape[0],cap_shape[1]],
                                  ],
                                 dtype=np.float32)


            self.M = cv2.getPerspectiveTransform(self.pts1,
                                                 self.pts2)
        else:
            self.M = None

            
        if self.cam_index is not None:
            self.cap = cv2.VideoCapture(self.cam_index)
        else:
            self.cap = None



    def capture_cam(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        else:
            frame = np.zeros( (256,256,3), dtype=np.uint8 )
            print(' - WARNING, cam is closed. ', file=sys.stderr)

        return frame


    def read_file(self):
        try:
            img = cv2.imread(self.img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            img = np.zeros( (256,256,3), dtype=np.uint8 )
            print(' - WARNING, unable to read image: {}.'.format(self.img_path), file=sys.stderr)

        return img
    

    def _raw_capture(self):
        if self.cap is None:
            frame = self.read_file()
        else:
            frame = self.capture_cam()

        return frame

    def frame_post_proc(self, frame, do_canny=True, do_close=True):
        if self.M is None:
            frame = cv2.resize(frame, self.cap_shape)
        else:
            frame = cv2.warpPerspective(frame,
                                        self.M,
                                        self.cap_shape)
        if do_canny:
            frame = cv2.Canny(frame, 100, 200)
        
            if do_close:
                frame = cv2.morphologyEx(frame,
                                         cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            frame = 255-frame
            
        return frame
        

    def capture(self):
        frame = self._raw_capture()
        frame = self.frame_post_proc(frame)
        return frame
            

    def test(self, fps=30, do_canny=True):
        raw_frame = self._raw_capture()

        self.windows = cv2_windows()

        self.windows.open_window(win_shape=(raw_frame.shape[1], raw_frame.shape[0]),
                                 win_name='Raw Capture')
     
        self.windows.open_window(win_shape=self.cap_shape,
                                 win_name='Proc Capture')
        

        
        while True:
            raw_frame  = self._raw_capture()
            proc_frame = self.frame_post_proc(raw_frame, do_canny)

            self.windows.update_img(img=raw_frame,
                                    win_name='Raw Capture')
            
            self.windows.update_img(img=proc_frame,
                                    win_name='Proc Capture')
            
            time.sleep(1/fps)

    def close(self):
        cam.windows.close_windows()
        if self.cap is not None:
            self.cap.release()
        
        return None
        

if __name__ == '__main__':
    
    
    cam = cam_capture(cam_index=None,
                      cap_shape=(256,300),
                      img_path='./test_img.jpeg')

    cam.test(do_canny=True)




    
