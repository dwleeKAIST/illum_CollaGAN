import requests
from requests.auth import HTTPDigestAuth
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import cv2
import base64
import numpy as np
import sys
from ipdb import set_trace as st

# Enabled since server's certificate is not authorized by CA.
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


#API_path = 'http://143.248.30.140:80/api-BISPL/denoise'
#auth = HTTPDigestAuth('user1', 'passwd')

### This is the part for loading input image
### in the form of base64 encoded string
#jpg_name = 'noise_img.jpeg'

ip       = sys.argv[1]
port     = sys.argv[2]
jpg_name = sys.argv[3]
#jpg_nameB = sys.argv[4]
#jpg_nameC = sys.argv[5]
#jpg_nameD = sys.argv[6]

API_path = 'http://'+ip+':'+port+'/api-BISPL/illum'

####
import codecs
with codecs.open('./illum/cat.png', encoding='base64_codec') as f:
    img_tmp = f.read()
    img_b64enc = base64.b64encode(img_tmp)
####
s = requests.Session()
for i in range(1):
    r = s.post(API_path, #auth=auth,
           headers={'Content-Type': 'application/octet-stream'}, params={'type':'img_illum'}, data=img_b64enc, verify=False)

#print(r.status_code)
#print(r.headers)

if r.status_code == 200 and r.headers['Content-Type'] == 'application/octet-stream':
    #
    img_noise = cv2.imread(jpg_name, cv2.IMREAD_COLOR)
    cv2.imshow('input : img with noise', img_noise)
    #
    img_data = base64.b64decode(r.content)
    img_byte = bytearray()
    img_byte.extend(img_data)
    
    jpg_bytes = np.asarray(img_byte, dtype=np.uint8)
    img = cv2.imdecode(jpg_bytes, cv2.IMREAD_UNCHANGED)
    cv2.imshow('output : img - denoised', img)
    cv2.waitKey()
else:
    print(r.content)

