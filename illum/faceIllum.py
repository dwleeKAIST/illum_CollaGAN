#import os
import numpy as np
from illum.Illum_net import Illum_net 
import tensorflow as tf
from scipy import misc
from skimage.color import rgb2ypbpr
from illum.util import wpng, apply_std
#from data.multipie import face as myDB
from ipdb import set_trace as st

class faceIllum:
    def __init__(self,ckpt_dir):
        self.ckpt_dir = ckpt_dir
        self.top = 0#3
        self.bottom = 0#3
        self.right = 0#3
        self.left = 0#3
        self.dtype = tf.float32
        str_ = "/device:GPU:0"
        with tf.device(str_):
            self.model = Illum_net()
        self.latest_ckpt = tf.train.latest_checkpoint( self.ckpt_dir )
        
  
    def do_illum(self, A,B,D,E):
        # check whether the img size X, Y are odd or not
#        self.checkWHodd(input_img)
        sz_ = A.shape
        mz = np.zeros([1,1,sz_[0],sz_[1]],dtype=np.float32)
        mo = np.ones( [1,1,sz_[0],sz_[1]],dtype=np.float32)
        mask = np.concatenate((mz,mz,mo,mz,mz),axis=1)
        
        Ay, stdA = self.my_y2r(A)
        By, stdB = self.my_y2r(B)
        Cy      = np.zeros([1,3,sz_[0],sz_[1]],dtype=np.float32)
        Dy, stdD = self.my_y2r(D)
        Ey, stdE = self.my_y2r(E)
        std = (stdA+stdB+stdD+stdE)/4
    
        with tf.Session() as sess:
            msg = []
            if not self.latest_ckpt==None:
                tf.train.Saver().restore(sess,self.latest_ckpt)
            else:
                print(self.ckpt_dir)
                msg = "There is no ckpt file for illumination compensation. Check the path again."
                return msg, []
 
            feed_dict = {self.model.a_img_Y:Ay, self.model.b_img_Y:By, self.model.c_img_Y:Cy, self.model.d_img_Y:Dy, self.model.e_img_Y:Ey,
                    self.model.mask:mask, self.model.is_Training:True}
            illum_img_Y = sess.run( self.model.recon_contrast_Y, feed_dict=feed_dict)
            illum_img = apply_std(illum_img_Y,std)
            
            return msg, illum_img

    def my_y2r(self, img_rgb, std=1.):
        img_y = rgb2ypbpr(img_rgb)
        if std==1.:
            std = np.std(img_y)
        img_y = img_y/std
        return np.swapaxes( img_y[np.newaxis,np.newaxis,:,:,:],1,4)[:,:,:,:,0], std

    def checkWHodd(self, img):
        sz = img.shape
        if sz[0]%2==1:
            self.bottom = self.bottom + 1
        if sz[1]%2==1:
            self.right  = self.right  + 1

    def padImg(self, img):
        return np.pad(img, pad_width=((self.top,self.bottom),(self.left,self.right),(0,0)),mode='edge')

    def cropImg(self, img):
        return img[self.top:-self.bottom, self.left:-self.right,:]

if __name__=="__main__":
    lpath = './illum/'
    tmp = faceIllum('./current_net/illum/DBrenew2_onlyFace_Unet_ssim1_ngf64_Dorig_Notresid/ckpt_dir')
    A = misc.imread(lpath+'116_01_01_051_02.png')
    B = misc.imread(lpath+'116_01_01_051_04.png')
    D = misc.imread(lpath+'116_01_01_051_10.png')
    E = misc.imread(lpath+'116_01_01_051_12.png')
     
    msg, rec = tmp.do_illum(A,B,D,E)
    wpng('./temp_recC.png', rec)


