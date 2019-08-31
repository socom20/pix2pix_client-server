import sys, os
import io
from PIL import Image

import threading
import signal
import ssl
from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer, SimpleSSLWebSocketServer
from optparse import OptionParser

import numpy as np

sys.path.append('../pix2pix_wrapper')
from pix2pix_wrapper import pix2pix_wrapper


##class SimpleEcho(WebSocket):
##    def handleMessage(self):
##        print('new msg:', self.data, type(self.data))
##        self.sendMessage(self.data)
##
##    def handleConnected(self):
##        print (self.address, 'connected ...')
##
##    def handleClose(self):
##        print (self.address, 'closed session ...')
##
##
##
##clients = []
##class SimpleChat(WebSocket):
##
##    def handleConnected(self):
##        print (self.address, 'connected ...')
##        
##        for client in clients:
##            client.sendMessage(self.address[0] + ' - connected')
##            
##        clients.append(self)
##        return None
##
##
##    def handleMessage(self):
##        for client in clients:
##            if client != self:
##                client.sendMessage(self.address[0] + u' - ' + self.data)
##
##
##    def handleClose(self):
##        clients.remove(self)
##        print (self.address, 'closed')
##        for client in clients:
##            client.sendMessage(self.address[0] + u' - disconnected')




clients = []
class Pix2Pix_predictor(WebSocket):

    def predict(self, img_bytes):
        img   = Image.open( io.BytesIO(img_bytes) )
        img_v = np.array(img).astype(np.float32)[np.newaxis,...] / 255.0

        img_pred = model.predict(img_v)[0]
        
        image    = Image.fromarray(img_pred)

        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()

        return imgByteArr

    
    def handleConnected(self):
        print(self.address, 'connected ...')
        
        for client in clients:
            client.sendMessage(self.address[0] + ' - connected')
            
        clients.append(self)
        return None


    def handleMessage(self):
        global data
        print(' - msg type={} from={}'.format(type(self.data), self.address))
        
        if type( self.data ) is bytearray:
            data = bytes(self.data)
            
            pred = self.predict(data)
            
            print(' - sending prediction ...')
            self.sendMessage( pred )

        elif type( self.data ) is str:
            self.sendMessage(' RC:' + self.data)


    def handleClose(self):
        clients.remove(self)
        print (self.address, 'closed')
        for client in clients:
            client.sendMessage(self.address[0] + u' - disconnected')



def start_new_server(server):
    server.serveforever()
    return None



class ws_server:
    def __init__(self, ws_class, host='localhost', port=8000, use_ssl=False, certfile='', keyfile=''):
        self.host = host
        self.port = port
        self.ws_class = ws_class
        self.use_ssl  = use_ssl

        self.certfile = certfile
        self.keyfile  = keyfile

        self.ssl_version = ssl.PROTOCOL_TLSv1

        self.server = None
        
        return None

        
    def start(self):
        if self.server is None:
            if not self.use_ssl:
                self.server = SimpleWebSocketServer(self.host,
                                                    self.port,
                                                    self.ws_class,
                                                    selectInterval=0.1)

            else:
                self.server = SimpleSSLWebSocketServer(self.host,
                                                       self.port,
                                                       self.ws_class,
                                                       self.certfile,
                                                       self.keyfile,
                                                       version=self.ssl_version,
                                                       selectInterval=0.1,
                                                       ssl_context=None)

            
            print(' - Starting WS Server, {}:{}'.format(self.host, self.port))
            
            self.th = threading.Thread(target=start_new_server, args=(self.server,))
            self.th.start()
            
        return None
    

    def close(self):
        if self.ws_server is not None:
            print(' - Closing WS Server ... Bye')
            self.ws_server.close()


        
if __name__ == "__main__":
    parser = OptionParser(usage="usage: %prog [options]", version="%prog 1.0")
    parser.add_option("--host", default='0.0.0.0', type='string', action="store", dest="host", help="hostname (localhost)")
    parser.add_option("--port", default=8000, type='int', action="store", dest="port", help="port (8000)")
    parser.add_option("--ssl", default=0, type='int', action="store", dest="ssl", help="ssl (1: on, 0: off (default))")
    parser.add_option("--cert", default='./cert.pem', type='string', action="store", dest="cert", help="cert (./cert.pem)")
    parser.add_option("--key", default='./key.pem', type='string', action="store", dest="key", help="key (./key.pem)")
    parser.add_option("--checkpoint", default='../pix2pix_wrapper/model_checkpoint', type='string', action="store", dest="checkpoint", help="a folder where the model checkpoint is located.")

    (options, args) = parser.parse_args()

    # Starting Predictor
    model = pix2pix_wrapper(options.checkpoint, load_model=True)

    cls = Pix2Pix_predictor
        
    server = ws_server(ws_class=cls,
                       host=options.host,
                       port=options.port,
                       use_ssl=options.ssl,
                       certfile=options.cert,
                       keyfile=options.key)
    
    server.start()

    def close_sig_handler(signal, frame):
        server.close()
        sys.exit()
        return None

    signal.signal(signal.SIGINT, close_sig_handler)








