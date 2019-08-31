import numpy as np
import cv2
import threading


class cv2_windows:
    
    def __init__(self):
        self.th = None
        self.keeprunning = False
        self.win_d = {}
        return None
    

    def open_window(self,
                    win_shape=(600,600),
                    win_name='Prediction'):
        
        if self.th is None:
            self.th = threading.Thread( target=self._win_update )
            self.keeprunning = True
            self.th.start()
        
        win_img = 128*np.ones( win_shape, dtype=np.uint8 )
        self.win_d[win_name] = [win_shape, win_img]

        return None


    def close_windows(self):
        self.keeprunning = False
        return None


    def _win_update(self):

        while self.keeprunning:
            for win_name, (win_shape, win_img) in self.win_d.items():
                cv2.imshow(win_name, win_img)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.keeprunning = False
        cv2.destroyAllWindows()
        self.th = None

        return None


    def update_img(self, img, win_name=None):
        """ the image needs to be in RGB mode """

        if win_name is None and len(self.win_d.keys()) == 1:
            win_name = list(self.win_d.keys())[0]
        elif win_name is None and len(self.win_d.keys()) > 1:
            raise Exception(' - ERROR, update_img: you need to specify a win_name if there are more than one windows opened.')
        
        if win_name not in self.win_d.keys():
            raise Exception(' - ERROR, update_img: bad window name.')
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        win_img = cv2.resize(img, tuple(self.win_d[win_name][0]) )
        self.win_d[win_name][1] = win_img
        
        return None


if __name__ == '__main__':
    win = cv2_windows()
    win.open_window(win_shape=(600,600), win_name='Prediction')
    win.open_window(win_shape=(200,600), win_name='Prediction2')
    
