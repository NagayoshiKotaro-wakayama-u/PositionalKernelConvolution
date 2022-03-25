from warnings import filters
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten, AveragePooling2D, GlobalAveragePooling2D,Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg
from PIL import Image
from libs.layers import PConv2D, siteConv, siteDeconv, Encoder,Decoder,PKEncoder
from libs.util import cmap,SqueezedNorm,pool2d
import os
import pdb
import matplotlib.pyplot as plt
import functools
from tensorflow.python.ops import nn, nn_ops, array_ops
from copy import deepcopy
import sys
import cv2

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# wrapper Model
## 各手法で共通の設定を行う
class InpaintingModel(tf.keras.Model):
    def __init__(self, img_rows=512, img_cols=512, loss_weights=[1,6,0.1], existOnly=False, exist_point_file=""):
        super(InpaintingModel, self).__init__()
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.loss_weights = loss_weights
        self.exist_only = existOnly
        # 存在する点が１・その他が０である入力と同サイズの画像を設定
        if exist_point_file=="":
            self.exist_img = np.ones([1,img_rows,img_cols,1])
        else:
            self.exist_img = np.array(Image.open(exist_point_file))[np.newaxis,:,:,np.newaxis]/255

        self.ones33 = K.constant(np.ones([3, 3, 1, 1]))
        self.tvLoss_conv_op = functools.partial(
            nn_ops.convolution_v2,
            strides=[1,1],
            padding='SAME',
            dilations=[1,1],
            data_format='NHWC',
            name='tvConv')


    def call(self):
        pass

    def compile(self, lr, mask):
        super().compile(
                optimizer = Adam(lr=lr),
                loss= self.loss_total(mask),
                metrics=[
                    self.loss_total(mask),
                    self.holeLoss(mask),
                    self.validLoss(mask),
                    self.tvLoss(mask),
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

        # pdb.set_trace()
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
        (masked_imgs, mask) = data
        return self((masked_imgs, mask), training=False)

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components
            # # 観測値部分の誤差
            l1 = self.validLoss(mask)(y_true, y_pred)
            # # 欠損部の誤差
            l2 = self.holeLoss(mask)(y_true, y_pred)
            
            # # 欠損部のまわり1pxの誤差
            l3 = self.tvLoss(mask)(y_true, y_pred)

            # # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            res = w1*l1 + w2*l2
            # total_loss = tf.add(w1*l1,w2*l2,name="loss_total")
            total_loss = tf.add(res,w3*l3,name="loss_total")
            return total_loss

        return lossFunction

    def holeLoss(self, mask):
        def loss_hole(y_true, y_pred):
            # 予測領域のみ損失を計算
            pred = y_pred*self.exist_img if self.exist_only else y_pred
            
            loss = self.l1((1-mask) * y_true, (1-mask) * pred)
            return loss
        return loss_hole

    def validLoss(self, mask):
        def loss_valid(y_true, y_pred):
            # 予測領域のみ損失を計算
            pred = y_pred*self.exist_img if self.exist_only else y_pred

            loss = self.l1(mask * y_true, mask * pred)
            return loss
        return loss_valid

    def tvLoss(self, mask):

        def loss_tv(y_true, y_pred):
            # 予測領域のみ損失を計算
            # pred = y_pred*self.exist_img if self.exist_only else y_pred
            pred = y_pred

            y_comp = mask * y_true + (1-mask) * pred
            # Create dilated hole region using a 3x3 kernel of all 1s.
            kernel = self.ones33

            dilated_mask = self.tvLoss_conv_op(1-mask, kernel)
            
            # Cast values to be [0., 1.], and compute dilated hole region of y_comp
            # pdb.set_trace()
            dilated_mask = K.cast_to_floatx(K.greater(dilated_mask, 0))
            P = dilated_mask * y_comp

            # Calculate total variation loss
            a = self.l1(P[:,1:,:,:], P[:,:-1,:,:])
            b = self.l1(P[:,:,1:,:], P[:,:,:-1,:])
            return a+b

        return loss_tv

    def PSNR(self,y_true, y_pred):
        # pdb.set_trace()
        exist = np.tile(self.exist_img,[y_pred.numpy().shape[0],1,1,1])
        pred = y_pred*exist
        mse = K.sum(K.square(pred - y_true)) / np.sum(exist)
        #return 20 * K.log(4.75) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.0)
        return - 10.0 * K.log(mse) / K.log(10.0)

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

#========================================
# PartialConv UNet（既存法）
## InpaintingUNetを継承し、モデルの構築・コンパイルのみ記述
class PConvUnetModel(InpaintingModel):
    def __init__(self, encStride=(2,2),**kwargs):
        super(PConvUnetModel, self).__init__(**kwargs)
        #========================================================
        # decide model
        self.encksize = [7,5,5,3,3]
        ## encoder
        self.encoder1 = Encoder(64, self.encksize[0], 1, strides=encStride, bn=False)
        self.encoder2 = Encoder(128,self.encksize[1], 2, strides=encStride)
        self.encoder3 = Encoder(256,self.encksize[2], 3, strides=encStride)
        self.encoder4 = Encoder(512,self.encksize[3], 4, strides=encStride) #TODO:元に戻す(512,3,4)
        self.encoder5 = Encoder(512,self.encksize[4], 5, strides=encStride) #TODO:元に戻す(512,3,5)

        ## decoder
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256,3)
        self.decoder8 = Decoder(128,3)
        self.decoder9 = Decoder(64,3)
        self.decoder10 = Decoder(3,3,bn=False)
        
        ## output
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        # self.ones33 = K.ones(shape=(3, 3, 1, 1))
        #========================================================

    def build_pconv_unet(self, masked, mask, training=True):
        e_conv1, e_mask1 = self.encoder1(masked,mask, istraining=training)
        e_conv2, e_mask2 = self.encoder2(e_conv1,e_mask1, istraining=training)
        e_conv3, e_mask3 = self.encoder3(e_conv2,e_mask2, istraining=training)
        e_conv4, e_mask4 = self.encoder4(e_conv3,e_mask3, istraining=training)
        e_conv5, e_mask5 = self.encoder5(e_conv4,e_mask4, istraining=training)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4, istraining=training)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3, istraining=training)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2, istraining=training)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1, istraining=training)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, masked, mask, istraining=training)

        outputs = self.conv2d(d_conv10)

        # TODO
        # 観測点を入力した値に置き換える
        # 周辺との連続性は損失でカバー
        
        return outputs

    def call(self, data, training=True):# data=(masked_imgs, mask)
        (masked_imgs, mask) = data
        output = self.build_pconv_unet(masked_imgs, mask, training=training)
        return output
    
    def predict_step(self, data):
        (masked_imgs, mask) = data
        return self((masked_imgs, mask), training=False)

#========================================
# PConv + 学習可能な位置特性
## 位置特性を学習によって取得する
## L1ノルム最小化でスパースにする
class PConvLearnSite(PConvUnetModel):
    def __init__(self, opeType='mul', obsOnlyL1 = False, fullLayer=False, siteLayers=[1], Lasso=False,  localLasso=False, **kwargs):
        super().__init__(**kwargs)
        # pdb.set_trace()
        self.phase = 0

        if len(siteLayers) != len(list(set(siteLayers))):
            # 重複がある場合はNG
            assert "siteLayers must be no duplication"

        if opeType=="add":
            self.initValue = np.zeros([1,self.img_rows,self.img_cols,1])
            self.scenter = 0
        elif opeType=="mul":
            self.initValue = np.ones([1,self.img_rows,self.img_cols,1])
            self.scenter = 1

        self.opeType = opeType
        self.obsOnlyL1 = obsOnlyL1
        self.localLasso = localLasso
        self.Lasso = Lasso
        if self.localLasso:
            self.lasso_conv = functools.partial(
                nn_ops.convolution_v2,
                strides=[1,1],
                padding="SAME",
                dilations=[1,1],
                data_format="NHWC",
                name="lassoConv")
            self.lasso_kernel = np.ones([16,16,1,1]).astype("float32")


        self.site_fullLayer = fullLayer
        # エンコーダの層数
        self.encLayerNum = 5
        # 位置特性を適用する層数(lnum)
        if fullLayer:
            self.siteLayers = [i+1 for i in range(self.encLayerNum)]
            self.lnum = self.encLayerNum
        else:
            self.siteLayers = siteLayers
            self.lnum = len(list(set(self.siteLayers)))
        
        # 入力層だけに学習可能な位置特性を設けている場合
        self.onlyInputSite = siteLayers[0] == 1 and self.lnum == 1
        if self.onlyInputSite:
            self.siteFeature = tf.Variable(
                self.initValue,
                trainable = True,
                name = "siteFeature-sparse",
                dtype = tf.float32
            )
        else:
            self.initValues = []
            self.siteFeatures = []
            self.exist_imgs = []
            # 位置特性の初期値・学習するパラメータを設定
            for lind in self.siteLayers:
                i = lind - 1
                size = [1,self.img_rows//(2**i),self.img_cols//(2**i),1]
                if opeType=="add":
                    init_v = np.zeros(size)
                elif opeType=="mul":
                    init_v = np.ones(size)

                self.initValues.append(init_v)
                self.siteFeatures.append(
                    tf.Variable(
                        init_v,
                        trainable = True,
                        name = f"siteFeature-sparse{i}",
                        dtype = tf.float32
                    )
                )
                # pdb.set_trace()
                resized_img = cv2.resize(
                    self.exist_img[0,:,:,0],
                    dsize=(
                        self.img_rows//(2**i),
                        self.img_cols//(2**i)
                    )
                )
                _, resized_img = cv2.threshold(resized_img,0.5,1,cv2.THRESH_BINARY)
                self.exist_imgs.append(resized_img)

    def build_pconv_unet(self, masked, mask, training=True, extract=""):
        # 位置特性の適用
        def applySite(x,i):
            # その層に位置特性
            if i in self.siteLayers:
                wind = self.siteLayers.index(i)
                _s = self.siteFeature if self.onlyInputSite else self.siteFeatures[wind]
                _s = K.tile(_s,[x.shape[0],1,1,x.shape[-1]])
                if self.opeType=="add":
                    res = x + _s
                elif self.opeType=="mul":
                    res = x * _s
            else:
                res = x

            return res

        # pdb.set_trace()
        e_conv1, e_mask1 = self.encoder1(applySite(masked,1), mask, istraining=training)
        e_conv2, e_mask2 = self.encoder2(applySite(e_conv1,2), e_mask1, istraining=training)
        e_conv3, e_mask3 = self.encoder3(applySite(e_conv2,3), e_mask2, istraining=training)
        e_conv4, e_mask4 = self.encoder4(applySite(e_conv3,4), e_mask3, istraining=training)
        e_conv5, e_mask5 = self.encoder5(applySite(e_conv4,5), e_mask4, istraining=training)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4, istraining=training)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3, istraining=training)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2, istraining=training)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1, istraining=training)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, masked, mask, istraining=training)

        outputs = self.conv2d(d_conv10)

        if extract != "":
            exec(f"outputs = {extract}")
                
        return outputs

    def encode_mask(self, mask):
        e_conv1, e_mask1 = self.encoder1(mask, mask, istraining=False)
        e_conv2, e_mask2 = self.encoder2(e_conv1, e_mask1, istraining=False)
        e_conv3, e_mask3 = self.encoder3(e_conv2, e_mask2, istraining=False)
        e_conv4, e_mask4 = self.encoder4(e_conv3, e_mask3, istraining=False)
        _, e_mask5 = self.encoder5(e_conv4, e_mask4, istraining=False)
        return [e_mask1,e_mask2,e_mask3,e_mask4,e_mask5]

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components
            # # 観測値部分の誤差
            l1 = self.validLoss(mask)(y_true, y_pred)
            # # 欠損部の誤差
            l2 = self.holeLoss(mask)(y_true, y_pred)
            
            # # 欠損部のまわり1pxの誤差
            l3 = self.tvLoss(mask)(y_true, y_pred)

            # # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            res = w1*l1 + w2*l2
            total_loss = tf.add(res,w3*l3,name="loss_total")
            return total_loss

        return lossFunction

    def compile(self, lr, mask):
        # InpaintingModelの親でコンパイル
        super(InpaintingModel, self).compile(
                optimizer = Adam(lr=lr),
                loss= self.loss_total(mask),
                metrics=[
                    self.loss_total(mask),
                    self.holeLoss(mask),
                    self.validLoss(mask),
                    self.tvLoss(mask),
                    self.PSNR,
                ],
                run_eagerly=True
            )
   
    def plotSiteFeature(self, epoch=None, plotSitePath="",saveName="",existmask=True):
        # pdb.set_trace()
        for i,sind in enumerate(self.siteLayers):

            if self.onlyInputSite:
                _s = np.squeeze(self.siteFeature.numpy())
                exist = self.exist_img[0,:,:,0]
            else:
                _s = np.squeeze(self.siteFeatures[i].numpy())
                exist = self.exist_imgs[i]
                
            plt.clf()
            plt.close()
            smax = np.max(_s)
            smin = np.min(_s)

            if existmask:
                _s[exist==0] = -10000
            # cmbwr = plt.get_cmap('viridis')
            cmbwr = plt.get_cmap('bwr')
            cmbwr.set_under('black')

            plt.imshow(_s,cmap=cmbwr,norm=SqueezedNorm(vmin=smin,vmax=smax,mid=self.scenter),interpolation='none')
            plt.colorbar(extend='both')
            if saveName == "":
                saveName = f"siteFeature{epoch}"
                
            plt.savefig(f"{plotSitePath}{os.sep}{saveName}_layer{sind}.png")
    
    def changeTrainPhase(self, phase):
        self.phase = phase

    def train_step(self, data):# data=(masked_imgs, mask, gt_imgs)
        (masked_imgs, mask, gt_imgs) = data[0]
        
        # 勾配の取得
        with tf.GradientTape() as tape:
            pred_imgs = self((masked_imgs, mask), training=True)
            loss = self.compiled_loss(gt_imgs, pred_imgs)

        # 勾配によるパラメータの更新
        trainable_vars = self.trainable_variables
        # pdb.set_trace()

        if self.phase == 1 or self.phase == 3: # 位置特性の学習なし
            trainable_vars = [
                v for v in trainable_vars if ('siteFeature' not in  v.name) and ('site_conv' not in v.name)
            ]
        elif self.phase == 2: # 位置特性と位置特性のConvのみ学習
            trainable_vars = [
                v for v in trainable_vars if ('siteFeature' in  v.name) or ('site_conv' in v.name)
            ]

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 評価値の更新
        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def getSiteFeature(self):
        if self.onlyInputSite:
            return self.siteFeature.numpy()
        else:
            return [s.numpy() for s in self.siteFeatures]

#========================================
# PKConv(最初の層のみ) + 学習可能な位置特性
## 位置特性を学習によって取得し、カーネルに乗算OR加算
## L1 ノルム最小化でスパースにする

class PKConvLearnSite(InpaintingModel):
    def __init__(self, opeType='mul', obsOnlyL1 = False, **kwargs):
        super().__init__(**kwargs)
        self.phase = 0

        if opeType=="add":
            self.initValue = np.zeros([1,self.img_rows,self.img_cols,1])
            self.scenter = 0

        elif opeType=="mul":
            self.initValue = np.ones([1,self.img_rows,self.img_cols,1])
            self.scenter = 1

        self.opeType = opeType
        self.obsOnlyL1 = obsOnlyL1
        self.siteFeature = tf.Variable(
            self.initValue,
            trainable = True,
            name = "siteFeature-sparse",
            dtype = tf.float32
        )

        #========================================================
        # decide model
        ## encoder
        self.encoder1 = PKEncoder(64, 7, opeType=opeType, bn=False)
        self.encoder2 = Encoder(128, 5, 2)
        self.encoder3 = Encoder(256, 5, 3)
        self.encoder4 = Encoder(512, 3, 4)
        self.encoder5 = Encoder(512, 3, 5)

        ## decoder
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256, 3)
        self.decoder8 = Decoder(128, 3)
        self.decoder9 = Decoder(64, 3)
        self.decoder10 = Decoder(3, 3, bn=False)

        ## output
        self.conv2d = Conv2D(1, 1, activation='sigmoid', name='output_img')

    def build_pkconv_unet(self, masked, mask, training=True):
        e_conv1, e_mask1 = self.encoder1(masked, mask, self.siteFeature, istraining=training)
        e_conv2, e_mask2 = self.encoder2(e_conv1, e_mask1, istraining=training)
        e_conv3, e_mask3 = self.encoder3(e_conv2, e_mask2, istraining=training)
        e_conv4, e_mask4 = self.encoder4(e_conv3, e_mask3, istraining=training)
        e_conv5, e_mask5 = self.encoder5(e_conv4, e_mask4, istraining=training)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4, istraining=training)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3, istraining=training)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2, istraining=training)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1, istraining=training)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, masked, mask, istraining=training)

        outputs = self.conv2d(d_conv10)

        return outputs

    def loss_total(self, mask):
        """
        Creates a loss function which sums all the loss components
        and multiplies by their weights. See paper eq. 7.
        """
        def lossFunction(y_true, y_pred):
            # Compute predicted image with non-hole pixels set to ground truth
            # Compute loss components
            # # 観測値部分の誤差
            l1 = self.validLoss(mask)(y_true, y_pred)
            # # 欠損部の誤差
            l2 = self.holeLoss(mask)(y_true, y_pred)
            
            # # 欠損部のまわり1pxの誤差
            l3 = self.tvLoss(mask)(y_true, y_pred)

            # # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            res = w1*l1 + w2*l2
            total_loss = tf.add(res,w3*l3,name="loss_total")
            return total_loss

        return lossFunction

    def compile(self, lr, mask):
        # InpaintingModelの親でコンパイル
        super(InpaintingModel, self).compile(
                optimizer = Adam(lr=lr),
                loss= self.loss_total(mask),
                metrics=[
                    self.loss_total(mask),
                    self.holeLoss(mask),
                    self.validLoss(mask),
                    self.tvLoss(mask),
                    self.PSNR
                ],
                run_eagerly=True
            )
   
    def plotSiteFeature(self, epoch=None, plotSitePath=""):
        plt.clf()
        plt.close()
        _s = self.siteFeature.numpy()
        smax = np.max(_s)
        smin = np.min(_s)
        # pdb.set_trace()
        _s[self.exist_img==0] = -10000
        cmbwr = plt.get_cmap('bwr')
        cmbwr.set_under('black')

        plt.imshow(_s[0,:,:,0],cmap=cmbwr,norm=SqueezedNorm(vmin=smin,vmax=smax,mid=self.scenter))
        plt.colorbar(extend='both')
        plt.savefig(f"{plotSitePath}{os.sep}siteFeature{epoch}.png")
        plt.close()
        
    def call(self, data, training=True):# data=(masked_imgs, mask)
        (masked_imgs, mask) = data
        output = self.build_pkconv_unet(masked_imgs, mask, training=training)
        return output

    def changeTrainPhase(self, phase):
        self.phase = phase

    def train_step(self, data):# data=(masked_imgs, mask, gt_imgs)
        (masked_imgs, mask, gt_imgs) = data[0]
        
        # 勾配の取得
        with tf.GradientTape() as tape:
            pred_imgs = self((masked_imgs, mask), training=True)
            loss = self.compiled_loss(gt_imgs, pred_imgs)

        # 勾配によるパラメータの更新
        trainable_vars = self.trainable_variables

        if self.phase == 1 or self.phase == 3: # 位置特性の学習なし
            trainable_vars = [
                v for v in trainable_vars if ('siteFeature' not in  v.name) and ('site_conv' not in v.name)
            ]
        elif self.phase == 2: # 位置特性と位置特性のConvのみ学習
            trainable_vars = [
                v for v in trainable_vars if ('siteFeature' in  v.name) or ('site_conv' in v.name)
            ]

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 評価値の更新
        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def getSiteFeature(self):
        return self.siteFeature.numpy()

#=====================================
############################################
class branchInpaintingModel(InpaintingModel):

    # 位置特性マップ・位置特性CNNの定義まで
    def __init__(self, opeType='add', siteLayers=[1], nonNeg=False, sConvActive=None, sCNNNum=3, use_softmax2D=True, **kwargs):
        super().__init__(**kwargs)

        if len(siteLayers) != len(list(set(siteLayers))):
            # 重複がある場合はNG
            assert "siteLayers must be no duplication"

        self.phase = 0
        self.siteLayers = siteLayers
        self.nonNegative = nonNeg
        sconv_constraint = NonNeg() if nonNeg else None

        self.channels = [1,64,128,256,512,512]
        self.encksize = [7,5,5,3,3]
        # TODO:現在は一つの階層にした位置特性を導入できない
        sconvChannel = self.channels[siteLayers[0]-1]

        # 位置特性のConvolution
        self.sCNNNum = sCNNNum
        
        ### 最終層は活性化なし
        sconvActivations = [
            None if i == sCNNNum-1 else sConvActive for i in range(sCNNNum)
        ]
        ### チャネル数
        sconvChannels = [
            sconvChannel//2 if i==0 else sconvChannel for i in range(sCNNNum)
        ]
        ### 位置特性のConv層
        self.siteConvs = [
            siteConv(
                sconvChannels[i], (3,3), strides=(1,1), activation=sconvActivations[i],
                padding='same', use_bias=False, kernel_constraint=sconv_constraint
            ) for i in range(sCNNNum)
        ]

        self.opeType = opeType
        if "add" in opeType or opeType=="mul":
            self.scenter = 0

        self.initValues = []
        self.siteFeatures = []
        self.exist_imgs = []
        # 位置特性の初期値・学習するパラメータを設定
        for lind in self.siteLayers:
            i = lind - 1
            size = [1,self.img_rows//(2**i),self.img_cols//(2**i),1]
            # 乗算も加算もどちらでも初期値は0
            if "add" in opeType or opeType=="mul":
                init_v = np.zeros(size)

            self.initValues.append(init_v)
            self.siteFeatures.append(
                tf.Variable(
                    init_v,
                    trainable = True,
                    name = f"siteFeature-sparse{i}",
                    dtype = tf.float32
                )
            )
            # pdb.set_trace()
            resized_img = cv2.resize(
                self.exist_img[0,:,:,0],
                dsize=(
                    self.img_rows//(2**i),
                    self.img_cols//(2**i)
                )
            )
            _, resized_img = cv2.threshold(resized_img,0.5,1,cv2.THRESH_BINARY)
            self.exist_imgs.append(resized_img)

    def getSiteFeature(self):
        return [_s.numpy() for _s in self.siteFeatures]

    def changeTrainPhase(self, phase):
        self.phase = phase

    def train_step(self, data):# data=(masked_imgs, mask, gt_imgs)
        (masked_imgs, mask, gt_imgs) = data[0]
        
        # 勾配の取得
        with tf.GradientTape() as tape:
            pred_imgs = self((masked_imgs, mask), training=True)
            loss = self.compiled_loss(gt_imgs, pred_imgs)

        # 勾配によるパラメータの更新
        trainable_vars = self.trainable_variables
        # pdb.set_trace()

        if self.phase == 1 or self.phase == 3: # 位置特性の学習なし
            trainable_vars = [
                v for v in trainable_vars if ('siteFeature' not in  v.name) and ('site_conv' not in v.name)
            ]
        elif self.phase == 2: # 位置特性と位置特性のConvのみ学習
            trainable_vars = [
                v for v in trainable_vars if ('siteFeature' in  v.name) or ('site_conv' in v.name)
            ]
        elif self.phase == 4: # 位置特性マップのみ
            trainable_vars = [
                v for v in trainable_vars if 'siteFeature' in  v.name
            ]
        elif self.phase == 5: # 位置特性Convのみ学習
            trainable_vars = [
                v for v in trainable_vars if 'site_conv' in v.name
            ]

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 評価値の更新
        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def plotSiteFeature(self, epoch=None, plotSitePath="", saveName="", plotfmap=True, existmask=True, samescale=False):

        if saveName == "":
            saveName = f"siteFeature{epoch}"

        if plotfmap:
            # pdb.set_trace()
            for i,exist in zip(self.siteLayers,self.exist_imgs):
                wind = self.siteLayers.index(i)
                sf = self.siteFeatures[wind]
                plot_w, plot_h = [8,8]
                plot_chmax = plot_w * plot_h
                cmbwr = plt.get_cmap('bwr')
                # cmbwr.set_under('black')

                # 位置特性のCNN
                sconvOut = sf
                for sconv in self.siteConvs:
                    sconvOut = sconv(sconvOut)

                plotsconvOut = sconvOut[0,:,:,:plot_chmax].numpy()
                sf = sf.numpy()[0,:,:,0]

                # スケールを同じにする場合
                if samescale:
                    smax = max([np.max(_s) for _s in [sf,plotsconvOut]])
                    smin = min([np.min(_s) for _s in [sf,plotsconvOut]])
                else:
                    smax = np.max(sf)
                    smin = np.min(sf)

                if existmask:
                    sf[exist==0] = -10000

                plt.clf()
                plt.close()
                plt.imshow(sf,cmap=cmbwr,norm=SqueezedNorm(vmin=smin,vmax=smax,mid=self.scenter),interpolation='none')
                plt.colorbar(extend='both')
                plt.savefig(f"{plotSitePath}{os.sep}{saveName}_layer3.png")

                plt.clf()
                plt.close()
                if not samescale:
                    smax = np.max(plotsconvOut)
                    smin = np.min(plotsconvOut)

                _, axes = plt.subplots(plot_h, plot_w, tight_layout=True, figsize=(plot_w*5, plot_h*5))
                for w in range(plot_w):
                    for h in range(plot_h):
                        axes[h,w].imshow(
                            plotsconvOut[:,:, w + h*plot_w],
                            cmap=cmbwr,
                            norm=SqueezedNorm(vmin=smin,vmax=smax,mid=self.scenter),
                            interpolation='none'
                        )
                        axes[h,w].set_title(f"ch {w + h*plot_w}")
                
                plt.savefig(f"{plotSitePath}{os.sep}{saveName}_sconv{len(self.siteConvs)}.png")



class branchPConv_lSite(branchInpaintingModel):
    def __init__(self, encStride=(2,2), **kwargs):
        super().__init__(**kwargs)
        # pdb.set_trace()
        self.phase = 0

        # decide model
        ## encoder
        self.encoder1 = Encoder(64, self.encksize[0], 1, strides=encStride, bn=False)
        self.encoder2 = Encoder(128,self.encksize[1], 2, strides=encStride)
        self.encoder3 = Encoder(256,self.encksize[2], 3, strides=encStride)
        self.encoder4 = Encoder(512,self.encksize[3], 4, strides=encStride) #TODO:元に戻す(512,3,4)
        self.encoder5 = Encoder(512,self.encksize[4], 5, strides=encStride) #TODO:元に戻す(512,3,5)

        ## decoder
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256,3)
        self.decoder8 = Decoder(128,3)
        self.decoder9 = Decoder(64,3)
        self.decoder10 = Decoder(3,3,bn=False)
        
        ## output
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')

    def build_pconv_unet(self, masked, mask, training=True):
        # 位置特性の適用
        def applySite(x,i):
            # 指定された層に位置特性を導入
            if (i in self.siteLayers):
                wind = self.siteLayers.index(i)
                _s = self.siteFeatures[wind]
                _s = K.tile(_s,[x.shape[0],1,1,1])
                
                # 位置特性マップを位置特性のCNNに通してチャネル数増大
                sconvOut = _s
                for sconv in self.siteConvs:
                    sconvOut = sconv(sconvOut)

                if self.phase == 1: # sconvのFreeze時は、PConvに影響を与えないようにする
                    sconvOut = sconvOut*0
                
                if self.opeType=="add":
                    res = x + sconvOut
                elif self.opeType=="mul": # 1 ＋ sconv2として乗算
                    res = x * (sconvOut+1)

            else: # i番目の層が位置特性を導入する場所でなかった場合
                res = x

            return res

        e_conv1, e_mask1 = self.encoder1(applySite(masked,1), mask, istraining=training)
        e_conv2, e_mask2 = self.encoder2(applySite(e_conv1,2), e_mask1, istraining=training)
        e_conv3, e_mask3 = self.encoder3(applySite(e_conv2,3), e_mask2, istraining=training)
        e_conv4, e_mask4 = self.encoder4(applySite(e_conv3,4), e_mask3, istraining=training)
        e_conv5, e_mask5 = self.encoder5(applySite(e_conv4,5), e_mask4, istraining=training)

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



class branchPKConv_lSite(branchInpaintingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ## encoder
        # siteLayersで指定されたEncoderを返す
        def applyPK(lind):
            bn = lind!=1 # 1層目のみbn=Falseとなる
            if lind in self.siteLayers:
                return PKEncoder(
                    self.channels[lind], 
                    self.encksize[lind-1], 
                    opeType=self.opeType,
                    bn=bn
                )
            else:
                return Encoder(self.channels[lind], self.encksize[lind-1], lind, bn=bn)

        self.encoder1 = applyPK(1)
        self.encoder2 = applyPK(2)
        self.encoder3 = applyPK(3)
        self.encoder4 = applyPK(4)
        self.encoder5 = applyPK(5)

        ## decoder
        self.decoder6 = Decoder(self.channels[4], 3)
        self.decoder7 = Decoder(self.channels[3], 3)
        self.decoder8 = Decoder(self.channels[2], 3)
        self.decoder9 = Decoder(self.channels[1], 3)
        self.decoder10 = Decoder(3, 3, bn=False)
        ## output
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        
    def build_pkconv_unet(self, masked, mask, training=True):
        
        def applySite(x1, x2, lind):
            encArgs=[x1, x2]
            if lind in self.siteLayers:
                wind = self.siteLayers.index(lind)
                sconvOut = self.siteFeatures[wind]
                for sconv in self.siteConvs:
                    sconvOut = sconv(sconvOut)

                if self.opeType=="mul":
                    sconvOut = sconvOut + 1

                encArgs.append(sconvOut)
            return encArgs

        # pdb.set_trace()
        e_conv1, e_mask1 = self.encoder1(*applySite(masked, mask, 1), istraining=training)
        e_conv2, e_mask2 = self.encoder2(*applySite(e_conv1, e_mask1, 2), istraining=training)
        e_conv3, e_mask3 = self.encoder3(*applySite(e_conv2, e_mask2, 3), istraining=training)
        e_conv4, e_mask4 = self.encoder4(*applySite(e_conv3, e_mask3, 4), istraining=training)
        e_conv5, e_mask5 = self.encoder5(*applySite(e_conv4, e_mask4, 5), istraining=training)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4, istraining=training)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3, istraining=training)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2, istraining=training)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1, istraining=training)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, masked, mask, istraining=training)

        outputs = self.conv2d(d_conv10)

        return outputs

    def call(self, data, training=True):# data=(masked_imgs, mask)
        (masked_imgs, mask) = data
        output = self.build_pkconv_unet(masked_imgs, mask, training=training)
        return output

