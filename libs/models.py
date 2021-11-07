import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from PIL import Image
from libs.layers import PConv2D, siteConv,Encoder,Decoder,PKEncoder
import os
import pdb
import matplotlib.pyplot as plt

## ラッパー
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNet
## 各手法共通の損失や設定を行う部分
class InpaintingUNet(object):
    def __init__(self, img_rows=512, img_cols=512, lr=0.0002,loss_weights=[1,6,0.1], inference_only=False, net_name='default', gpus=1):
        # Settings
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.lr = lr
        self.loss_weights=loss_weights
        # Set current epoch
        self.current_epoch = 0

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components

            # 観測値部分の誤差
            l1 = self.loss_valid(mask, y_true, y_pred)
            # 欠損部の誤差
            l2 = self.loss_hole(mask, y_true, y_pred)
            
            # 欠損部のまわり1pxの誤差
            y_comp = mask * y_true + (1-mask) * y_pred
            l3 = self.loss_tv(mask, y_comp)

            # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            res = w1*l1 + w2*l2

            total_loss = tf.add(res,w3*l3,name="loss_total")

            # Return loss function
            return total_loss
        return lossFunction

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = self.ones33
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])
        return a+b

    def fit_generator(self, generator, *args, **kwargs):
        """Fit the U-Net to a (images, targets) generator

        Args:
            generator (generator): generator supplying input image & mask, as well as targets.
            *args: arguments to be passed to fit_generator
            **kwargs: keyword arguments to be passed to fit_generator
        """
        self.model.fit_generator(
            generator,
            *args, **kwargs
        )

    def fit(self, **kwargs):
        self.model.fit(**kwargs)
        return


    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())

    def load(self, filepath, train_bn=True, lr=0.0002):
        # Load weights into model
        epoch = os.path.basename(filepath).split('.')[1].split('-')[0]
        try:
            epoch = int(epoch)
        except ValueError:
            self.current_epoch = 100
        else:
            self.current_epoch = epoch

        self.model.load_weights(filepath)

    # @staticmethod
    def PSNR(self,y_true, y_pred):
        pred = y_pred*self.exist_img
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(K.mean(K.square(pred - y_true))) / K.log(10.0)

    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# traceable Model
## 勾配の取得などが可能なModelクラス
class traceableModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(traceableModel, self).__init__(*args, **kwargs)

    def train_step(self, data):# data=((masked_imgs, mask), gt_imgs)
        (masked_imgs, mask, gt_imgs) = data[0]
        
        # 勾配の取得
        with tf.GradientTape() as tape:
            pred_imgs = self((masked_imgs, mask), training=True)
            loss = self.compiled_loss(gt_imgs, mask, pred_imgs)

        # 勾配によるパラメータの更新
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 評価値の更新
        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (masked_imgs, mask, gt_imgs) = data[0]

        pred_imgs = self((masked_imgs, mask), training=False)
        self.compiled_loss(gt_imgs, pred_imgs, regularization_losses=self.losses)

        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        return self(data, training=False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#========================================
# PartialConv UNet
## InpaintingUNetを継承し、モデルの構築・コンパイルのみ記述
class PConvUnet(InpaintingUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # decide model
        ## Input layer
        self.inputs_img = Input((self.img_rows, self.img_cols, 1), name='inputs_img')
        self.inputs_mask = Input((self.img_rows, self.img_cols, 1), name='inputs_mask')        

        ## encoder
        self.encoder1 = Encoder(64, 7, 1, bn=False)
        self.encoder2 = Encoder(128,5, 2)
        self.encoder3 = Encoder(256,5, 3)
        self.encoder4 = Encoder(512,3, 4) #TODO:元に戻す(512,3,4)
        self.encoder5 = Encoder(512,3, 5) #TODO:元に戻す(512,3,5)

        ## decoder        
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256,3)
        self.decoder8 = Decoder(128,3)
        self.decoder9 = Decoder(64,3)
        self.decoder10 = Decoder(3,3,bn=False)
        
        ## output
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        # self.ones33 = K.ones(shape=(3, 3, 1, 1))
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        


    def build_pconv_unet(self, train_bn=True):
        e_conv1, e_mask1 = self.encoder1(self.inputs_img,self.inputs_mask)
        e_conv2, e_mask2 = self.encoder2(e_conv1,e_conv1)
        e_conv3, e_mask3 = self.encoder3(e_conv2,e_conv2)
        e_conv4, e_mask4 = self.encoder4(e_conv3,e_conv3)
        e_conv5, e_mask5 = self.encoder5(e_conv4,e_conv4)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, self.inputs_img, self.inputs_mask)

        outputs = self.conv2d(d_conv10)
        
        return outputs
#========================================
# PositionalKernelConv UNet
## 位置依存な畳み込みを行うUNet
## siteCNNを内部に持つ
class PKConvUNet(InpaintingUNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #TODO:モデルの記述

#========================================
# siteCNN

#========================================


# sample
class PConvUnetModel(tf.keras.Model):
    def __init__(self):
        super(PConvUnetModel, self).__init__()
        self.loss_weights = [1,6,0.1]
        ## encoder
        self.encoder1 = Encoder(64, 7, 1, bn=False)
        self.encoder2 = Encoder(128,5, 2)
        self.encoder3 = Encoder(256,5, 3)
        self.encoder4 = Encoder(512,3, 4) #TODO:元に戻す(512,3,4)
        self.encoder5 = Encoder(512,3, 5) #TODO:元に戻す(512,3,5)

        ## decoder        
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256,3)
        self.decoder8 = Decoder(128,3)
        self.decoder9 = Decoder(64,3)
        self.decoder10 = Decoder(3,3,bn=False)
        
        ## output
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        # self.ones33 = K.ones(shape=(3, 3, 1, 1))

    def build_pconv_unet(self, masked, mask, training=True):
        e_conv1, e_mask1 = self.encoder1(masked,mask, istraining=training)
        e_conv2, e_mask2 = self.encoder2(e_conv1,e_conv1, istraining=training)
        e_conv3, e_mask3 = self.encoder3(e_conv2,e_conv2, istraining=training)
        e_conv4, e_mask4 = self.encoder4(e_conv3,e_conv3, istraining=training)
        e_conv5, e_mask5 = self.encoder5(e_conv4,e_conv4, istraining=training)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4, istraining=training)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3, istraining=training)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2, istraining=training)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1, istraining=training)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, masked, mask, istraining=training)

        outputs = self.conv2d(d_conv10)
        
        return outputs

    def call(self, data, training=True):# data=(masked_imgs, mask)
        (masked_imgs, mask) = data
        output = self.build_pconv_unet(masked_imgs, mask, training=training)
        return output

    def compile(self, lr, mask):
        super().compile(
                optimizer = Adam(lr=lr),
                loss= self.loss_total(mask),
                metrics=[
                    self.loss_total(mask),
                    self.PSNR
                ],
                run_eagerly=True
            )
        
    def train_step(self, data):# data=(masked_imgs, mask, gt_imgs)
        (masked_imgs, mask, gt_imgs) = data[0]
        
        # 勾配の取得
        with tf.GradientTape() as tape:
            pred_imgs = self((masked_imgs, mask), training=True)
            loss = self.compiled_loss(gt_imgs, pred_imgs)

        # 勾配によるパラメータの更新
        trainable_vars = self.trainable_variables
        # pdb.set_trace()
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 評価値の更新
        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (masked_imgs, mask, gt_imgs) = data[0]

        pred_imgs = self((masked_imgs, mask), training=False)
        # pdb.set_trace()
        # plt.clf()
        # plt.close()
        # plt.imshow(pred_imgs[0,:,:,0])
        # plt.colorbar()
        # plt.savefig("tmp_pred.png")
        # plt.clf()
        # plt.close()
        # plt.imshow(gt_imgs[0,:,:,0])
        # plt.colorbar()
        # plt.savefig("tmp_gt.png")

        self.compiled_loss(gt_imgs, pred_imgs, regularization_losses=self.losses)

        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        return self(data, training=False)

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components
            
            # total_loss = self.l1(y_true,y_pred)
            # # 観測値部分の誤差
            l1 = self.loss_valid(mask, y_true, y_pred)
            # # 欠損部の誤差
            l2 = self.loss_hole(mask, y_true, y_pred)
            
            # # 欠損部のまわり1pxの誤差
            # y_comp = mask * y_true + (1-mask) * y_pred
            # l3 = self.loss_tv(mask, y_comp)

            # # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            # res = w1*l1 + w2*l2
            total_loss = tf.add(w1*l1,w2*l2,name="loss_total")
            # total_loss = tf.add(res,w3*l3,name="loss_total")

            # Return loss function
            return total_loss
        return lossFunction

    def loss_hole(self, mask, y_true, y_pred):
        """Pixel L1 loss within the hole / mask"""
        return self.l1((1-mask) * y_true, (1-mask) * y_pred)

    def loss_valid(self, mask, y_true, y_pred):
        """Pixel L1 loss outside the hole / mask"""
        return self.l1(mask * y_true, mask * y_pred)

    def loss_tv(self, mask, y_comp):
        """Total variation loss, used for smoothing the hole region, see. eq. 6"""

        # Create dilated hole region using a 3x3 kernel of all 1s.
        kernel = self.ones33
        dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')
        
        # Cast values to be [0., 1.], and compute dilated hole region of y_comp
        dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
        P = dilated_mask * y_comp

        # Calculate total variation loss
        a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
        b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])
        return a+b

    def PSNR(self,y_true, y_pred):
        # pred = y_pred*self.exist_img
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)

    @staticmethod
    def l1(y_true, y_pred):
        """Calculate the L1 loss used in all loss calculations"""
        if K.ndim(y_true) == 4:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
        elif K.ndim(y_true) == 3:
            return K.mean(K.abs(y_pred - y_true), axis=[1,2])
        else:
            raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
