import tensorflow as tf
import tensorflow.contrib.layers as li
from ipdb import set_trace as st
from illum.netUtil import  Conv2d, Conv2d2x2, lReLU, BN, Conv1x1, Unet, Unet31, Unet_shallow, SRnet, tmpnet, SRnet31, SRnet2, UnetL3
from illum.util import tf_Y2R,tf_YC2R

dtype = tf.float32
d_form  = 'channels_first'
d_form_ = 'NCHW'
ch_dim  = 1

REAL_LABEL = 0.9
eps        = 1e-12
class Illum_net:
    def __init__(self):
        self.nB    = 1 #opt.nB
        self.nCh_in = 64 #opt.nCh_in
        self.nCh_out = 3 #opt.nCh_out
        self.nY    = 240 #opt.nY
        self.nX    = 240 #opt.nX
        self.lr    = 0.0001 #opt.lr
        self.lr_D  = 0.00001 #opt.lr_D
        self.lr_C  = 0.00001 #opt.lr_C
        self.nCh   = 64  #opt.ngf
        self.nCh_D = 4 #opt.nCh_D
        self.use_lsgan = True #opt.use_lsgan
        self.class_N = 5
        self.lambda_l1 = 0.0 #opt.lambda_l1
        self.lambda_l2 = 10.0 #opt.lambda_l2
        self.lambda_G_clsf = 1.0 #opt.lambda_G_clsf
        self.lambda_D_clsf = 1.0 #opt.lambda_D_clsf
        self.lambda_ssim = 1.0 #opt.lambda_ssim

        self.G = Generator('G', 'Unet', 1,nCh=self.nCh,use_1x1Conv=False, w_decay=0.01,resid=False)
        self.D = Discriminator('D', nCh=self.nCh_D, w_decay_D=0.0,class_N=self.class_N, DR_ratio=0.2)
        # placeholders 
        #self.targets_Y = tf.placeholder(dtype, [self.nB, self.nCh_out, self.nY, self.nX])
        #self.targets = tf_Y2R(self.targets_Y)
        self.tar_class_idx = tf.placeholder(tf.uint8)
        self.is_Training = tf.placeholder(tf.bool)
        
        self.a_img_Y = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])
        self.b_img_Y = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      
        self.c_img_Y = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])
        self.d_img_Y = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      
        self.e_img_Y = tf.placeholder(dtype,[self.nB,self.nCh_out, self.nY, self.nX])      

        self.a_img = tf_Y2R(self.a_img_Y)
        self.b_img = tf_Y2R(self.b_img_Y)
        self.c_img = tf_Y2R(self.c_img_Y)
        self.d_img = tf_Y2R(self.d_img_Y)
        self.e_img = tf_Y2R(self.e_img_Y)

        self.mask = tf.placeholder(dtype, [self.nB,5,self.nY,self.nX])

#        self.a_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
#        self.b_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
#        self.c_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
#        self.d_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
#        self.e_mask = tf.placeholder(dtype,[self.nB,1, self.nY, self.nX])
#
        self.bool0 = tf.placeholder(tf.bool)
        self.bool1 = tf.placeholder(tf.bool)
        self.bool2 = tf.placeholder(tf.bool)
        self.bool3 = tf.placeholder(tf.bool)
        self.bool4 = tf.placeholder(tf.bool)

        ''' generate inputs ( imag + mask ) '''
        #tmp_zeros = tf.zeros([self.nB,self.nCh_out,self.nY,self.nX],dtype)
        #inp1 = tf.cond(self.bool0, lambda:tmp_zeros, lambda:self.a_img_Y)
        #inp2 = tf.cond(self.bool1, lambda:tmp_zeros, lambda:self.b_img_Y)
        #inp3 = tf.cond(self.bool2, lambda:tmp_zeros, lambda:self.c_img_Y)
        #inp4 = tf.cond(self.bool3, lambda:tmp_zeros, lambda:self.d_img_Y)
        #inp5 = tf.cond(self.bool4, lambda:tmp_zeros, lambda:self.e_img_Y)
        tmp_zeros = tf.zeros([self.nB,self.nCh_out,self.nY,self.nX],dtype)
        inp1 = self.a_img_Y
        inp2 = self.b_img_Y
        inp3 = tmp_zeros
        inp4 = self.d_img_Y
        inp5 = self.e_img_Y     

        input_contrasts = tf.concat([inp1[:,0,tf.newaxis,:,:],inp2[:,0,tf.newaxis,:,:],inp3[:,0,tf.newaxis,:,:],inp4[:,0,tf.newaxis,:,:],inp5[:,0,tf.newaxis,:,:] ],axis=ch_dim) 
#        self.inputs = tf.concat([input_contrasts, self.a_mask, self.b_mask,self.c_mask,self.d_mask,self.e_mask],axis=ch_dim)
        self.inputs = tf.concat([input_contrasts, self.mask],axis=ch_dim)

        ''' inference G, D for 1st input (not cyc) '''
        recon_Y = self.G(self.inputs,self.is_Training)
        recon_CbCr = (inp1[:,1:,:,:]+inp2[:,1:,:,:]+inp3[:,1:,:,:]+inp4[:,1:,:,:]+inp5[:,1:,:,:])/4.
        self.recon_contrast_Y = tf.concat([recon_Y,recon_CbCr],axis=ch_dim) 
        self.recon_contrast = tf_YC2R(recon_Y, recon_CbCr)

        ## D(recon)
#        RealFake_rec,type_rec = self.D(self.recon_contrast, self.is_Training)
        ## D(target)
        #RealFake_tar,type_tar = self.D(self.targets, self.is_Training)
#        tmp_ones = tf.ones([self.nB,1,self.nY,self.nX],dtype)
        
        ''' generate inputs for cyc '''
        # for cyc
#        cyc1_ = tf.cond(self.bool0, lambda:recon_Y, lambda:self.a_img_Y[:,0,tf.newaxis,:,:])
#        cyc2_ = tf.cond(self.bool1, lambda:recon_Y, lambda:self.b_img_Y[:,0,tf.newaxis,:,:])
#        cyc3_ = tf.cond(self.bool2, lambda:recon_Y, lambda:self.c_img_Y[:,0,tf.newaxis,:,:])
#        cyc4_ = tf.cond(self.bool3, lambda:recon_Y, lambda:self.d_img_Y[:,0,tf.newaxis,:,:])
#        cyc5_ = tf.cond(self.bool4, lambda:recon_Y, lambda:self.e_img_Y[:,0,tf.newaxis,:,:])
#        
#        atmp_zeros = tf.zeros([self.nB,1,self.nY,self.nX],dtype)
#        cyc_inp1_ = tf.concat([atmp_zeros,cyc2_,cyc3_,cyc4_,cyc5_],axis=ch_dim)
#        cyc_inp2_ = tf.concat([cyc1_,atmp_zeros,cyc3_,cyc4_,cyc5_],axis=ch_dim)
#        cyc_inp3_ = tf.concat([cyc1_,cyc2_,atmp_zeros,cyc4_,cyc5_],axis=ch_dim)
#        cyc_inp4_ = tf.concat([cyc1_,cyc2_,cyc3_,atmp_zeros,cyc5_],axis=ch_dim)
#        cyc_inp5_ = tf.concat([cyc1_,cyc2_,cyc3_,cyc4_,atmp_zeros],axis=ch_dim)    
#
#        cyc_inp1 = tf.concat([cyc_inp1_,tmp_ones,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
#        cyc_inp2 = tf.concat([cyc_inp2_,atmp_zeros,tmp_ones,atmp_zeros,atmp_zeros,atmp_zeros],axis=ch_dim)
#        cyc_inp3 = tf.concat([cyc_inp3_,atmp_zeros,atmp_zeros,tmp_ones,atmp_zeros,atmp_zeros],axis=ch_dim)
#        cyc_inp4 = tf.concat([cyc_inp4_,atmp_zeros,atmp_zeros,atmp_zeros,tmp_ones,atmp_zeros],axis=ch_dim)
#        cyc_inp5 = tf.concat([cyc_inp5_,atmp_zeros,atmp_zeros,atmp_zeros,atmp_zeros,tmp_ones],axis=ch_dim)
#        
        ''' inference G, D for cyc inputs'''
#        cyc1_Y = self.G(cyc_inp1, self.is_Training)
#        cyc2_Y = self.G(cyc_inp2, self.is_Training)
#        cyc3_Y = self.G(cyc_inp3, self.is_Training)
#        cyc4_Y = self.G(cyc_inp4, self.is_Training)
#        cyc5_Y = self.G(cyc_inp5, self.is_Training)
#
#        self.cyc1_Y = tf.concat([cyc1_Y, recon_CbCr],axis=ch_dim)
#        self.cyc2_Y = tf.concat([cyc2_Y, recon_CbCr],axis=ch_dim)
#        self.cyc3_Y = tf.concat([cyc3_Y, recon_CbCr],axis=ch_dim)
#        self.cyc4_Y = tf.concat([cyc4_Y, recon_CbCr],axis=ch_dim)
#        self.cyc5_Y = tf.concat([cyc5_Y, recon_CbCr],axis=ch_dim)
# 
#        self.cyc1 = tf_YC2R(cyc1_Y, recon_CbCr)
#        self.cyc2 = tf_YC2R(cyc2_Y, recon_CbCr)
#        self.cyc3 = tf_YC2R(cyc3_Y, recon_CbCr) 
#        self.cyc4 = tf_YC2R(cyc4_Y, recon_CbCr)
#        self.cyc5 = tf_YC2R(cyc5_Y, recon_CbCr)
#        ## D(cyc), C(cyc)
#        RealFake_cyc1,type_cyc1 = self.D(self.cyc1, self.is_Training)
#        RealFake_cyc2,type_cyc2 = self.D(self.cyc2, self.is_Training)
#        RealFake_cyc3,type_cyc3 = self.D(self.cyc3, self.is_Training)
#        RealFake_cyc4,type_cyc4 = self.D(self.cyc4, self.is_Training)
#        RealFake_cyc5,type_cyc5 = self.D(self.cyc5, self.is_Training)
#        
#        ## D(tar), C(tar)
#        RealFake_tar1, type_tar1 = self.D(self.a_img, self.is_Training)
#        RealFake_tar2, type_tar2 = self.D(self.b_img, self.is_Training)
#        RealFake_tar3, type_tar3 = self.D(self.c_img, self.is_Training)
#        RealFake_tar4, type_tar4 = self.D(self.d_img, self.is_Training)
#        RealFake_tar5, type_tar5 = self.D(self.e_img, self.is_Training)
#
#        ''' Here, loss def starts here'''
        # gen loss for generator
#        G_gan_loss_cyc1   = tf.reduce_mean(tf.squared_difference(RealFake_cyc1, REAL_LABEL))
#        G_gan_loss_cyc2   = tf.reduce_mean(tf.squared_difference(RealFake_cyc2, REAL_LABEL))
#        G_gan_loss_cyc3   = tf.reduce_mean(tf.squared_difference(RealFake_cyc3, REAL_LABEL))
#        G_gan_loss_cyc4   = tf.reduce_mean(tf.squared_difference(RealFake_cyc4, REAL_LABEL))
#        G_gan_loss_cyc5   = tf.reduce_mean(tf.squared_difference(RealFake_cyc5, REAL_LABEL))
#        G_gan_loss_cyc  = G_gan_loss_cyc1 + G_gan_loss_cyc2 + G_gan_loss_cyc3 + G_gan_loss_cyc4 + G_gan_loss_cyc5
#        
#        G_gan_loss_orig   = tf.reduce_mean(tf.squared_difference(RealFake_rec, REAL_LABEL))
#        G_gan_loss = (G_gan_loss_orig + G_gan_loss_cyc)/6.
#
#
#        OH_label1 = tf.tile(tf.reshape(tf.one_hot(tf.cast(0,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
#        OH_label2 = tf.tile(tf.reshape(tf.one_hot(tf.cast(1,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
#        OH_label3 = tf.tile(tf.reshape(tf.one_hot(tf.cast(2,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
#        OH_label4 = tf.tile(tf.reshape(tf.one_hot(tf.cast(3,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
#        OH_label5 = tf.tile(tf.reshape(tf.one_hot(tf.cast(4,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
#        OH_labelT = tf.tile(tf.reshape(tf.one_hot(tf.cast(self.tar_class_idx,tf.uint8),self.class_N),[-1,1,self.class_N]),[self.nB,1,1])
#       
#        '''classification loss for generator'''
#        G_clsf_cyc_loss1 = tf.losses.softmax_cross_entropy(OH_label1, type_cyc1)
#        G_clsf_cyc_loss2 = tf.losses.softmax_cross_entropy(OH_label2, type_cyc2)
#        G_clsf_cyc_loss3 = tf.losses.softmax_cross_entropy(OH_label3, type_cyc3)
#        G_clsf_cyc_loss4 = tf.losses.softmax_cross_entropy(OH_label4, type_cyc4)
#        G_clsf_cyc_loss5 = tf.losses.softmax_cross_entropy(OH_label5, type_cyc5)
#        G_clsf_cyc_loss  = G_clsf_cyc_loss1 + G_clsf_cyc_loss2 + G_clsf_cyc_loss3 + G_clsf_cyc_loss4 + G_clsf_cyc_loss5
#
#        G_clsf_orig_loss = tf.losses.softmax_cross_entropy(OH_labelT, type_rec)
#        G_clsf_loss = (G_clsf_orig_loss + G_clsf_cyc_loss)/6.
#        
#        # discriminator loss
#        C_loss1 = tf.losses.softmax_cross_entropy(OH_label1,type_tar1)
#        C_loss2 = tf.losses.softmax_cross_entropy(OH_label2,type_tar2)
#        C_loss3 = tf.losses.softmax_cross_entropy(OH_label3,type_tar3)
#        C_loss4 = tf.losses.softmax_cross_entropy(OH_label4,type_tar4)
#        C_loss5 = tf.losses.softmax_cross_entropy(OH_label5,type_tar5)
#        self.C_loss       = C_loss1 + C_loss2 + C_loss3 + C_loss4 + C_loss5
#        
#        if self.use_lsgan:
#            #err_real = tf.reduce_mean(tf.squared_difference(RealFake_tar, REAL_LABEL))
#            err_fake = tf.reduce_mean(tf.square(RealFake_rec))
#            #D_err = err_real + err_fake
#            D_err = err_fake
#
#            cyc_real1 = tf.reduce_mean(tf.squared_difference(RealFake_tar1, REAL_LABEL))
#            cyc_fake1 = tf.reduce_mean(tf.square(RealFake_cyc1))
#            cyc_err1 = cyc_real1 + cyc_fake1 
#            cyc_real2 = tf.reduce_mean(tf.squared_difference(RealFake_tar2, REAL_LABEL))
#            cyc_fake2 = tf.reduce_mean(tf.square(RealFake_cyc2))
#            cyc_err2 = cyc_real2 + cyc_fake2 
#            cyc_real3 = tf.reduce_mean(tf.squared_difference(RealFake_tar3, REAL_LABEL))
#            cyc_fake3 = tf.reduce_mean(tf.square(RealFake_cyc3))
#            cyc_err3 = cyc_real3 + cyc_fake3 
#            cyc_real4 = tf.reduce_mean(tf.squared_difference(RealFake_tar4, REAL_LABEL))
#            cyc_fake4 = tf.reduce_mean(tf.square(RealFake_cyc4))
#            cyc_err4 = cyc_real4 + cyc_fake4 
#            cyc_real5 = tf.reduce_mean(tf.squared_difference(RealFake_tar5, REAL_LABEL))
#            cyc_fake5 = tf.reduce_mean(tf.square(RealFake_cyc5))
#            cyc_err5 = cyc_real5 + cyc_fake5
#        else:
#            st()
#            #err_real = -tf.reduce_mean(tf.log(RealFake_tar+eps))
#            err_fake = -tf.reduce_mean(tf.log(1-RealFake_rec+eps))
#        D_gan_cyc  = cyc_err1 + cyc_err2 + cyc_err3 + cyc_err4 + cyc_err5
#        D_gan_loss  = (D_err + D_gan_cyc)/6.
#        self.D_loss = (D_err + D_gan_cyc)/6. + (self.C_loss)/5.
#
#       
#        self.cyc1_rgb = self.tf_vis(self.cyc1)
#        self.cyc2_rgb = self.tf_vis(self.cyc2)
#        self.cyc3_rgb = self.tf_vis(self.cyc3)
#        self.cyc4_rgb = self.tf_vis(self.cyc4)
#        self.cyc5_rgb = self.tf_vis(self.cyc5)
       
#        self.a_img_rgb = self.tf_vis( self.a_img )
#        self.b_img_rgb = self.tf_vis( self.b_img )
#        self.c_img_rgb = self.tf_vis( self.c_img )
#        self.d_img_rgb = self.tf_vis( self.d_img )
#        self.e_img_rgb = self.tf_vis( self.e_img )
#

    def tf_vis(self, inp, order=[0,2,3,1],scale=40.):
        return tf.cast( tf.transpose(inp,order)*scale,tf.uint8)
    def tf_vis_abs(self, inp, order=[0,2,3,1],scale=40.):
        return tf.cast( tf.transpose( tf.abs(inp),order)*scale,tf.uint8)




    def optimize(self, G_loss, D_loss, C_loss):
        def make_optimizer(loss, variables, lr,  name='Adam'):
            global_step = tf.Variable(0,trainable=False)
            decay_step  = 400
            lr_         = tf.train.exponential_decay(lr, global_step, decay_step,0.99,staircase=True)
            tf.summary.scalar('learning_rate/{}'.format(name), lr_)
            return tf.train.AdamOptimizer( lr_, beta1=0.5 , name=name).minimize(loss,global_step=global_step,var_list=variables)
        
        self.G_optm  = make_optimizer(G_loss, self.G.variables, self.lr,   name='Adam_G')
        self.D_optm  = make_optimizer(D_loss, self.D.variables, self.lr_D, name='Adam_D')
        #self.C_optm  = make_optimizer(C_loss, self.C.variables, self.lr_C, name='Adam_C')
        self.C_optm  = make_optimizer(C_loss, self.D.variables, self.lr_C, name='Adam_C')

class Generator:
    def __init__(self,name,G, nCh_out,nCh=16, use_1x1Conv=False, w_decay=0, resid=False):
        if G=='Unet' or G=='Unet1x1':
            self.net = Unet
        elif G=='Unet31':
            self.net = Unet31
        elif G=='SRnet31':
            self.net = SRnet31
        elif G=='SRnet1x1' or G=='SRnet':
            self.net = SRnet
        elif G=='SRnet2':
            self.net = SRnet2
        elif G=='UnetL3':
            self.net = UnetL3
        else:
            st()
        self.name = name
        self.nCh  = nCh
        self.nCh_out = nCh_out
        self.reuse = False
        self.use_1x1Conv=use_1x1Conv
        self.w_decay = w_decay 
        self.resid = resid
        #(image, ch_out, is_Training,str_direction='AtoB', nCh=64):
    '''
    str_direction : AtoB / BtoA
    '''
    def __call__(self, image, is_Training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            reg_ = tf.contrib.layers.l2_regularizer(scale=self.w_decay) if self.w_decay>0 else None 
            #out = self.net(image, self.nCh_out, is_Training, reg_, nCh=self.nCh)
            if self.resid:
                ref = (image[:,0,tf.newaxis,:,:]+image[:,1,tf.newaxis,:,:]+image[:,2,tf.newaxis,:,:]+image[:,3,tf.newaxis,:,:]+image[:,4,tf.newaxis,:,:])/4.
                out = ref +  self.net(image,self.nCh_out, is_Training, reg_, nCh=self.nCh, _1x1Conv=self.use_1x1Conv)

                #out = ref+(1/self.beta*tf.log(1+tf.exp(self.beta*
                #    self.net(image,self.nCh_out, is_Training, reg_, nCh=self.nCh,
                #        _1x1Conv=self.use_1x1Conv))))
            else: 
                out = self.net(image, self.nCh_out, is_Training, reg_, nCh=self.nCh, _1x1Conv=self.use_1x1Conv)        
            #out = Unet(image,self.nCh_out, is_Training, reg_, nCh=self.nCh)
            #out = Unet_shallow(image,self.nCh_out,is_Training, reg_,nCh=self.nCh,use_1x1Conv=self.use_1x1Conv)
            #out = SRnet(image,self.nCh_out,is_Training, [],nCh=self.nCh,use_1x1Conv=self.use_1x1Conv)
            #out = SRnet31(image,self.nCh_out,is_Training, reg_,nCh=self.nCh)

        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return out#, logits

class Discriminator:
    def __init__(self, name='D', nCh=4, w_decay_D=0,DR_ratio=0,class_N=5):
        self.name   = name
        self.nCh    = [nCh, int(nCh*2), int(nCh*4),int(nCh*8)]#, int(nCh*16), int(nCh*32), int(nCh*64)]
        #self.nCh    = [nCh, nCh, nCh,nCh, nCh,nCh, nCh]
        self.reuse  = False
        self.k = 4
        self.kernel = 240/(2**(len(self.nCh)-1))
#        self.kernel = 512/(2**3)
        self.w_decay_D = w_decay_D
        self.dropout_ratio = DR_ratio
        self.class_N=class_N

    def __call__(self, input, is_Training):
        with tf.variable_scope(self.name, reuse=self.reuse):
            reg_ = tf.contrib.layers.l2_regularizer(scale=self.w_decay_D) if self.w_decay_D>0 else None

            ## img size = 240
            str_= self.name+'1D'
            h0 = lReLU( Conv2d2x2(input, kernel_size=self.k, ch_out= self.nCh[0], reg=reg_, name=str_), name=str_)
            ##img size = 120
            str_= self.name+'2D'
            h1 = lReLU( Conv2d2x2(   h0, kernel_size=self.k, ch_out=self.nCh[1], reg=reg_, name=str_), name=str_)
            ##img size = 60
            str_= self.name+'3D'
            h2 = lReLU( Conv2d2x2(   h1, kernel_size=self.k, ch_out=self.nCh[2], reg=reg_, name=str_), name=str_)
            ###img size = 30
            str_= self.name+'4D'
            hLast = lReLU( Conv2d2x2(   h2, kernel_size=self.k, ch_out=self.nCh[3], reg=reg_, name=str_), name=str_)
            ##img size = 15

            #str_= self.name+'5D'
            #h4 = lReLU( Conv2d2x2(   h3, kernel_size=self.k, ch_out=self.nCh[4], reg=reg_, name=str_), name=str_)
            ###img size = 
            #str_= self.name+'6D'
            #h5 = lReLU( Conv2d2x2(   h4, kernel_size=self.k, ch_out=self.nCh[5], reg=reg_, name=str_), name=str_)
            ####img size = 8x8
            #str_= self.name+'7D'
            #hLast = lReLU( Conv2d2x2(   h5, kernel_size=self.k, ch_out=self.nCh[6], reg=reg_, name=str_), name=str_)
            ####img size = 4x4
            #str_= self.name+'LastD'
            hLast = tf.layers.dropout(hLast, rate=self.dropout_ratio,training=is_Training)

            RF_out = tf.layers.conv2d(hLast,filters=1,kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=False, data_format=d_form, kernel_initializer=li.xavier_initializer(), name='RF_conv')
            #logits = tf.layers.dense(tf.contrib.layers.flatten(hLast),self.class_N, use_bias=False, name='LastD_class')
        self.reuse=True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return RF_out,[[[1.,0.,0.,0.,0.]]]#, logits[tf.newaxis,:,:]


