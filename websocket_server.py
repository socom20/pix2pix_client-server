import sys, os
import io
from PIL import Image

import threading
import signal
import ssl
from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer, SimpleSSLWebSocketServer
from optparse import OptionParser

import numpy as np
import json

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

        if len(img_v.shape) == 3:
            # New axis for the color channel
            img_v = np.repeat(img_v[...,np.newaxis], 3, axis=-1)

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
        print(' - msg_type={} from={}'.format(type(self.data), self.address))

        try:
            if 'pass_ok' not in dir(self):
                self.pass_ok = False
                
            if self.pass_ok == False:
                if ws_server.password is None:
                    # There is no password
                    self.pass_ok = True
                    
                elif type(self.data) is str:
                    # This msg is forvalidateing the password
                    if self.data == ws_server.password:
                        self.pass_ok = True
                        print(' - handleMessage: PassWord OK!!')
                        
                        e_s = json.dumps({'msg': 'PassWord OK!!'})
                        self.sendMessage( e_s )

                        # Nothing more to do!!!
                        return None
                    
                    else:
                        print(' - handleMessage: BaD PassWord', file=sys.stderr)
                    
        
            if self.pass_ok:
                # Everything is ok, lets interpretate the msg
                if type( self.data ) is bytearray:
                    data = bytes(self.data)
                    print(' Makeing prediction ...')
                    try:
                        pred = self.predict(data)
                    except Exception as e:
                        e_d = {'error': str(e), 'where':'predict'}
                        e_s = json.dumps(e_d)
                        print("WARNING:", e, file=sys.stderr)
                        self.sendMessage( e_s )
                        return None
                        
                    print(' - sending prediction ...')
                    self.sendMessage( pred )

                elif type( self.data ) is str:
                    self.sendMessage(' RC:' + self.data)
                
            else:
                # Everything is not ok, connection refused!!!
                # Password rejected
                e_d = {'error': 'critical ;)', 'where':'handleMessage'}
                e_s = json.dumps(e_d)
                self.sendMessage( e_s )
                self.close()


        except Exception as e:
            e_d = {'error': str(e), 'where':'handleMessage: unhandled'}
            e_s = json.dumps(e_d)
            self.sendMessage( e_s )
            print(' - ERROR, handleMessage: unhandled:', e, file=sys.stderr)
    
        return None

    def handleClose(self):
        clients.remove(self)
        print (self.address, 'closed')
        for client in clients:
            client.sendMessage(self.address[0] + u' - disconnected')



def start_new_server(server):
    try:
        server.serveforever()
    except ValueError as e:
        pass
    except Exception as e:
        raise e
    
    return None



class ws_server:
    def __init__(self,
                 ws_class,
                 host='localhost',
                 port=8000,
                 use_ssl=False,
                 certfile='',
                 keyfile='',
                 password='rtypopuioghj951435dsads'):
        
        self.host = host
        self.port = port
        self.ws_class = ws_class
        self.use_ssl  = use_ssl

        self.certfile = certfile
        self.keyfile  = keyfile
        ws_server.password = password


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
        if self.server is not None:
            print(' - Closing WS Server ... Bye')
            self.server.close()
            self.server = None


        
if __name__ == "__main__":
    parser = OptionParser(usage="usage: %prog [options]", version="%prog 1.0")
    parser.add_option("--host", default='0.0.0.0', type='string', action="store", dest="host", help="hostname (localhost)")
    parser.add_option("--port", default=8080, type='int', action="store", dest="port", help="port (80)")
    parser.add_option("--password", default=None, type='str', action="store", dest="password", help="server password")
    parser.add_option("--ssl", default=0, type='int', action="store", dest="ssl", help="ssl (1: on, 0: off (default))")
    parser.add_option("--cert", default='./cert.pem', type='string', action="store", dest="cert", help="cert (./cert.pem)")
    parser.add_option("--key", default='./key.pem', type='string', action="store", dest="key", help="key (./key.pem)")
    parser.add_option("--checkpoint", default='../pix2pix_wrapper/model_checkpoint', type='string', action="store", dest="checkpoint", help="a folder where the model checkpoint is located.")


    (options, args) = parser.parse_args(['--password', 'sergio'])

    # Starting Predictor
    model = pix2pix_wrapper(options.checkpoint, load_model=True)

    cls = Pix2Pix_predictor
        
    server = ws_server(ws_class=cls,
                       host=options.host,
                       port=options.port,
                       password=options.password,
                       use_ssl=options.ssl,
                       certfile=options.cert,
                       keyfile=options.key)
    
    server.start()

    def close_sig_handler(signal, frame):
        server.close()
        sys.exit()
        return None

    signal.signal(signal.SIGINT, close_sig_handler)








