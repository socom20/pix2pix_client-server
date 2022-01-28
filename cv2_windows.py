import numpy as np
import cv2
import threading

def click_event(event, x, y, flags, params):

    if event==cv2.EVENT_LBUTTONDOWN:
        print(' Mouse Position = [{}, {}]'.format(x, y))

    return None


class cv2_windows:
    
    def __init__(self, print_mouse_pos=False):
        self.th = None
        self.keeprunning = False
        self.win_d = {}
        self.fullscreen = False

        self.set_mouse_event = print_mouse_pos
        return None
    

    def open_window(self,
                    win_shape=(600,600),
                    win_name='Prediction'):
        
        if self.th is None:
            self.th = threading.Thread( target=self._win_update )
            self.keeprunning = True
            self.lock = threading.Lock()
            self.th.start()
        
        win_img = 128*np.ones( win_shape, dtype=np.uint8 )
        with self.lock:
            self.win_d[win_name] = [win_shape, win_img]

        return None


    def close_windows(self):
        self.keeprunning = False
        return None


    def _win_update(self):
            
        while self.keeprunning:
            with self.lock:
                for win_name, (win_shape, win_img) in self.win_d.items():
                    cv2.imshow(win_name, win_img)

                if self.set_mouse_event and len(self.win_d.items()) > 0:
                    for win_name, (win_shape, win_img) in self.win_d.items():
                        print('Setting:', win_name)
                        cv2.setMouseCallback(win_name, click_event)
                        
                    self.set_mouse_event = False
                
            wkey = cv2.waitKey(10) & 0xFF
            if wkey == ord('q'):
                break
            
            elif wkey == ord('f'):
                self.fullscreen = not self.fullscreen
                cv2.destroyAllWindows()
                
                if self.fullscreen:
                    for win_name, (win_shape, win_img) in self.win_d.items():
                        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
                        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.keeprunning = False
        cv2.destroyAllWindows()
        self.th = None

        return None


    def update_img(self, img, win_name=None):
        """ the image needs to be in RGB mode """
        
        with self.lock:
            if win_name is None and len(self.win_d.keys()) == 1:
                win_name = list(self.win_d.keys())[0]
                
            elif win_name is None and len(self.win_d.keys()) > 1:
                raise Exception(' - ERROR, update_img: you need to specify a win_name if there are more than one windows opened.')
        
            if win_name not in self.win_d.keys():
                raise Exception(' - ERROR, update_img: bad window name.')

        if len(img.shape) == 2:
            img = np.repeat(img[...,np.newaxis], 3, axis=-1)
            
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        win_img = cv2.resize(img, tuple(self.win_d[win_name][0][::-1]) )
        
        with self.lock:
            self.win_d[win_name][1] = win_img
        
        return None


if __name__ == '__main__':
    win = cv2_windows(True)
    win.open_window(win_shape=(500,700), win_name='Prediction')
##    win.open_window(win_shape=(200,600), win_name='Prediction2')

    img = cv2.imread('test_img.jpeg')[:,:,::-1]
    win.update_img(img)
