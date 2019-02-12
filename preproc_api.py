import cherrypy as cp
import numpy as np
import base64
from illum.faceIllum import faceIllum

import cv2
from ipdb import set_trace as st
import shutil, os, io
#from SISR.faceSISR import faceSISR
import time

class FileLenIO(io.FileIO):
    def __init__(self, name, mode='r', closefd=True):
        io.FileIO.__init__(self, name, mode, closefd)
        self.__size = statinfo = os.stat(name).st_size
    def __len__(self):
        return self.__size

class IllumCore:
    _store  = None
    exposed = True
    
    def __init__(self):
        self.core = faceIllum(ckpt_dir='./current_net/illum/DBrenew2_onlyFace_Unet_ssim1_ngf64_Dorig_Notresid')
        
    def GET(self):
        return "illum core module"

    def POST(self, type):
        img_data = cp.request.body.read(int(cp.request.headers['Content-Length']))
        img_data = base64.b64decode(img_data)
        img_byte = bytearray()
        img_byte.extend(img_data)
        
        
        jpg_bytes = np.asarray(img_byte, dtype=np.uint8)
        img = cv2.imdecode(jpg_bytes, cv2.IMREAD_UNCHANGED)
        
        sz = img.shape[1]
        A = img[:,:sz,:]
        B = img[:,sz:2*sz,:]
        D = img[:,2*sz:3*sz,:]
        E = img[:,3*sz:,:]

        [msg, img_out] = self.core.do_illum(A,B,D,E)
        
        r, buf = cv2.imencode('.jpg', img_out)
        b = base64.b64encode(buf.tobytes())

        response = cp.response
        response.status = '200 OK'
        response.headers['Content-Type'] = 'application/octet-stream'

        dict = {'a':100, 'type':type}
        return b

