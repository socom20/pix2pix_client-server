import numpy as np
import cv2
import threading


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


if __name__ == '__main__':
    win = cv2_window(win_shape=(600,600), win_name='Prediction')
    win.open_window()
    
