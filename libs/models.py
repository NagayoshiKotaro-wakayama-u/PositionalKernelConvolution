from warnings import filters
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input,Conv2D, Dense, Flatten, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
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
## 各手法で共通の設定を行う部分
## 勾配などの追跡が容易なモデル
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
# PartialConv UNet
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
# PositionalKernelConv UNet
## 位置依存な畳み込みを行うUNet
## siteCNNを内部に持つ
class PKConvUNetAndSiteCNN(InpaintingModel):
    def __init__(self, opeType='mul', siteInputChan=1,siteConvF=[1,1,1,1], site_range=[0.1,1],
        siteSigmoid=False, siteClip=False, learnMultiSiteW=False, UNet_pretrain=0, SCNN_pretrain=0, siteSinglePath=True, 
        useSiteGAP = False, **kwargs):
        super().__init__(**kwargs)

        #========================================================
        # decide model
                
        # site CNNの設定
        ## siteCNNの最終出力が常に用いられるかどうか
        self.singlePath = siteSinglePath
        self.useSiteGAP = useSiteGAP
        if siteSinglePath:
            if useSiteGAP:
                siteConvF = [12,48,12,siteInputChan]
                self.siteGAP = GlobalAveragePooling2D()
                siteInputChan = 1
            else:
                siteInputChan = 1
                siteConvF = [12,48,12,siteInputChan]

            ## site CNN
            self.sCNN1 = siteConv(filters=siteConvF[0], strides=(2,2), kernel_size=(5,5), padding='same')
            self.sCNN2 = siteConv(filters=siteConvF[1], strides=(2,2), kernel_size=(3,3), padding='same')

            self.sCNN3 = siteDeconv(filters=siteConvF[2], strides=(1,1), kernel_size=(3,3), padding='same')
            self.sCNN4 = siteDeconv(filters=siteConvF[3], strides=(1,1), kernel_size=(5,5), padding='same')
            self.avePool = AveragePooling2D()
        else:
            ## site CNN
            self.sCNN1 = siteConv(filters=siteConvF[0], strides=(2,2), kernel_size=(7,7), padding='same')
            self.sCNN2 = siteConv(filters=siteConvF[1], strides=(2,2), kernel_size=(5,5), padding='same')
            self.sCNN3 = siteConv(filters=siteConvF[2], strides=(2,2), kernel_size=(5,5), padding='same')
            self.sCNN4 = siteConv(filters=siteConvF[3], strides=(2,2), kernel_size=(3,3), padding='same')
        
        ## encoder
        ### 1層目の引数
        encKeyArgs = {
            "siteInputChan":siteInputChan,
            "site_range":site_range,
            "siteSigmoid":siteSigmoid,
            "siteClip":siteClip,
            "opeType":opeType,
            "learnMultiSiteW":learnMultiSiteW,
            "bn":False}
        
        ### 2階層目から一部の引数が変化する 
        encKeyArgsList = [deepcopy(encKeyArgs) for _ in range(4)]            
        for i in range(4):
            if siteSinglePath:
                encKeyArgsList[i].update({"bn":True})
            else:
                encKeyArgsList[i].update({
                    "siteInputChan":siteConvF[i],
                    "bn":True
                })
        
        self.encoder1 = PKEncoder(64, 7, **encKeyArgs)
        self.encoder2 = PKEncoder(128, 5, **encKeyArgsList[0])
        self.encoder3 = PKEncoder(256, 5, **encKeyArgsList[1])
        self.encoder4 = PKEncoder(512, 3, **encKeyArgsList[2])
        self.encoder5 = PKEncoder(512, 3, **encKeyArgsList[3])

        ## decoder
        self.decoder6 = Decoder(512, 3)
        self.decoder7 = Decoder(256,3)
        self.decoder8 = Decoder(128,3)
        self.decoder9 = Decoder(64,3)
        self.decoder10 = Decoder(3,3,bn=False)

        ## output
        self.conv2d = Conv2D(1,1,activation='sigmoid',name='output_img')
        #========================================================

        # UNetの事前学習
        self.UNet_pretrain = UNet_pretrain
        self.UNet_pretrainCount = 0

        # 位置特性CNNの事前学習
        self.SCNN_pretrain = SCNN_pretrain
        self.sCNN_pretrainCount = 0

    def build_pkconv_unet(self, masked, mask, site, training=True, plotSitePath=""):

        site_conv1 = self.sCNN1(site)
        site_conv2 = self.sCNN2(site_conv1)
        site_conv3 = self.sCNN3(site_conv2)
        site_conv4 = self.sCNN4(site_conv3)

        if self.useSiteGAP:
            site_weight = self.siteGAP(site_conv4)
            site_weight = tf.expand_dims(tf.expand_dims(site_weight,1),1)
            site_weight = tf.tile(site_weight,[1,self.img_rows,self.img_cols,1])
            site_conv4 = tf.math.reduce_sum(site_conv4*site_weight,axis=-1,keepdims=True,name="squeezed_site")
            
        if self.singlePath:
            # シングルパスの場合
            e_conv1, e_mask1 = self.encoder1(masked, mask, site_conv4, istraining=training)

            half_site = self.avePool(site_conv4)
            e_conv2, e_mask2 = self.encoder2(e_conv1, e_mask1, half_site, istraining=training)

            half_site = self.avePool(half_site)
            e_conv3, e_mask3 = self.encoder3(e_conv2, e_mask2, half_site, istraining=training)
    
            half_site = self.avePool(half_site)
            e_conv4, e_mask4 = self.encoder4(e_conv3, e_mask3, half_site, istraining=training)

            half_site = self.avePool(half_site)
            e_conv5, e_mask5 = self.encoder5(e_conv4, e_mask4, half_site, istraining=training)
        else:
            # マルチパスの場合
            e_conv1, e_mask1 = self.encoder1(masked, mask, site, istraining=training)
            e_conv2, e_mask2 = self.encoder2(e_conv1, e_mask1, site_conv1, istraining=training)
            e_conv3, e_mask3 = self.encoder3(e_conv2, e_mask2, site_conv2, istraining=training)
            e_conv4, e_mask4 = self.encoder4(e_conv3, e_mask3, site_conv3, istraining=training)
            e_conv5, e_mask5 = self.encoder5(e_conv4, e_mask4, site_conv4, istraining=training)

        d_conv6, d_mask6 = self.decoder6(e_conv5, e_mask5, e_conv4, e_mask4, istraining=training)
        d_conv7, d_mask7 = self.decoder7(d_conv6, d_mask6, e_conv3, e_mask3, istraining=training)
        d_conv8, d_mask8 = self.decoder8(d_conv7, d_mask7, e_conv2, e_mask2, istraining=training)
        d_conv9, d_mask9 = self.decoder9(d_conv8, d_mask8, e_conv1, e_mask1, istraining=training)
        d_conv10, _ = self.decoder10(d_conv9, d_mask9, masked, mask, istraining=training)

        outputs = self.conv2d(d_conv10)

        #=============================================
        # pdb.set_trace()
        if plotSitePath != "":
            sites = [site_conv1,site_conv2,site_conv3,site_conv4]
            sites = [_s.numpy()[0] for _s in sites]
            chans = [_s.shape[-1] for _s in sites]
            for i,_s in enumerate(sites):
                # _,axes = plt.subplots(1,4,figsize=(18,5))
                for ch in range(chans[i]):
                #     axes[ch].set_title(f"channel {ch+1}")
                #     axes[ch].imshow(cmap(_s[:,:,ch]))
                    plt.clf()
                    plt.close()
                    plt.imshow(_s[:,:,ch])
                    plt.title(f"channel {ch+1}")
                    plt.colorbar()
                    plt.savefig(f"{plotSitePath}{os.sep}layer{i+1}_ch{ch+1}_site.png")

        # pdb.set_trace()
        #=============================================

        return outputs

    def call(self, data, training=True):# data=(masked_imgs, mask)
        (masked_imgs, mask, site) = data
        output = self.build_pkconv_unet(masked_imgs, mask, site, training=training)
        return output

    def train_step(self, data):# data=(masked_imgs, mask, site, gt_imgs)
        (masked_imgs, mask, site, gt_imgs) = data[0]
        
        # 勾配の取得
        with tf.GradientTape() as tape:
            pred_imgs = self((masked_imgs, mask, site), training=True)
            loss = self.compiled_loss(gt_imgs, pred_imgs)

        # 勾配によるパラメータの更新
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # パラメータを更新
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 位置特性CNNの事前学習
        if self.SCNN_pretrain > self.sCNN_pretrainCount:
            ## site CNNのパラメータを抜き出す
            siteParams,siteGrads = [],[]
            for i,v in enumerate(trainable_vars):
                if 'site_conv' in v.name.split("/")[-2]:
                    siteParams.append(v)
                    siteGrads.append(gradients[i])
            
            self.optimizer.apply_gradients(zip(siteGrads, siteParams))
            self.sCNN_pretrainCount += 1
        else:
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 評価値の更新
        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        (masked_imgs, mask, site, gt_imgs) = data[0]
        pred_imgs = self((masked_imgs, mask, site), training=False)

        self.compiled_loss(gt_imgs, pred_imgs, regularization_losses=self.losses)

        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        (masked_imgs, mask, site) = data
        return self((masked_imgs, mask, site), training=False)

#========================================
# concat-PartialConv UNet
## 位置特性をチャネル方向に結合する
class chConcatPConvUNet(PConvUnetModel):
    def call(self, data, training=True):# data=(masked_imgs, mask)
        (masked_imgs, mask) = data
        mask = np.tile(mask,[1,1,1,masked_imgs.shape[3]])
        output = self.build_pconv_unet(masked_imgs, mask, training=training)
        return output
    
#========================================
# PConv + 学習可能な位置特性
## 位置特性を学習によって取得する
## L1ノルム最小化でスパースにする

class PConvLearnSite(PConvUnetModel):
    def __init__(self, opeType='mul', obsOnlyL1 = False, fullLayer=False, siteLayers=[1], localLasso=False, **kwargs):
        super().__init__(**kwargs)
        # pdb.set_trace()

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

    def LocalLasso_site(self, mask):

        if self.onlyInputSite:
            siteDelta = self.siteFeature - self.initValue
            siteDelta = siteDelta * siteDelta # 二乗誤差
        else:
            siteDeltas = [self.siteFeatures[i] - self.initValues[i] for i,_ in enumerate(self.siteLayers)]
            siteDeltas = [d*d for d in siteDeltas] # 二乗誤差
            # エンコーダ各階層のマスク画像のリスト
            # pdb.set_trace()
            emasks = self.encode_mask(mask)
            masks = []

            for lind in self.siteLayers:
                if lind==1:
                    masks.append(mask)
                else:
                    masks.append(emasks[lind-2][:,:,:,:1])

        def localLasso_loss(y_true, y_pred):
            if self.onlyInputSite:# 入力層のみ学習可能な位置特性がある場合                
                if self.obsOnlyL1:
                    lasso_mask = self.lasso_conv(mask, self.lasso_kernel)
                    lasso_mask = K.sign(lasso_mask, 0, 1)
                    lasso_img = self.lasso_conv(siteDelta*mask, self.lasso_kernel)
                    res = K.sum(K.abs(lasso_img*lasso_mask))/K.sum(lasso_mask)
                else:
                    lasso_img = self.lasso_conv(siteDelta, self.lasso_kernel)
                    res = K.mean(lasso_img)
            else:
                res = 0
                for i,_ in enumerate(self.siteLayers):
                    if self.obsOnlyL1:
                        # pdb.set_trace()
                        lasso_mask = self.lasso_conv(masks[i], self.lasso_kernel)
                        lasso_mask = K.sign(lasso_mask)
                        lasso_img = self.lasso_conv(siteDeltas[i]*masks[i], self.lasso_kernel)
                        res = K.sum(K.abs(lasso_img*lasso_mask))/K.sum(lasso_mask)
                    else:
                        lasso_img = self.lasso_conv(siteDeltas[i], self.lasso_kernel)
                        res = K.mean(lasso_img)
            return res
        return localLasso_loss

    def L1_site(self, mask):

        # pdb.set_trace()
        if self.onlyInputSite:
            siteDelta = self.siteFeature - self.initValue
        else:
            siteDeltas = [self.siteFeatures[i] - self.initValues[i] for i,_ in enumerate(self.siteLayers)]
            # エンコーダ各階層のマスク画像のリスト
            emasks = self.encode_mask(mask)
            masks = []

            for lind in self.siteLayers:
                if lind==1:
                    masks.append(mask)
                else:
                    masks.append(emasks[lind-2][:,:,:,:1])
        
        def loss_L1site(y_true, y_pred):

            if self.onlyInputSite:# 入力層のみ学習可能な位置特性がある場合
                if self.obsOnlyL1:
                    res = K.sum(K.abs(siteDelta*mask))/K.sum(mask)
                else:
                    res = K.mean(siteDelta)
            
            else:

                res = 0
                for i,_ in enumerate(self.siteLayers):
                    if self.obsOnlyL1:
                        res = K.sum(K.abs(siteDeltas[i]*masks[i]))/K.sum(masks[i])
                    else:
                        res = K.mean(siteDeltas[i])

            return res
        return loss_L1site

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

            if self.localLasso:
                l1SiteLoss = self.LocalLasso_site(mask)(y_true, y_pred)
            else:
                l1SiteLoss = self.L1_site(mask)(y_true, y_pred)

            # # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            res = w1*l1 + w2*l2 + l1SiteLoss
            # total_loss = tf.add(w1*l1,w2*l2,name="loss_total")
            total_loss = tf.add(res,w3*l3,name="loss_total")
            return total_loss

        return lossFunction

    def compile(self, lr, mask):
        met = [
            self.loss_total(mask),
            self.holeLoss(mask),
            self.validLoss(mask),
            self.tvLoss(mask),
            self.L1_site(mask),
            self.PSNR,
        ]
        if self.localLasso:
            met.append(self.LocalLasso_site(mask))
            
        # InpaintingModelの親でコンパイル
        super(InpaintingModel, self).compile(
                optimizer = Adam(lr=lr),
                loss= self.loss_total(mask),
                metrics=met,
                run_eagerly=True
            )
   
    def plotSiteFeature(self, epoch=None, plotSitePath="",saveName=""):
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

            _s[exist==0] = -10000
            # cmbwr = plt.get_cmap('viridis')
            cmbwr = plt.get_cmap('bwr')
            cmbwr.set_under('black')

            plt.imshow(_s,cmap=cmbwr,norm=SqueezedNorm(vmin=smin,vmax=smax,mid=self.scenter),interpolation='none')
            plt.colorbar(extend='both')
            if saveName == "":
                saveName = f"siteFeature{epoch}"
                
            plt.savefig(f"{plotSitePath}{os.sep}{saveName}_layer{sind}.png")
    
    # テスト用
    def setSiteFeature(self, site):
        if self.onlyInputSite:
            self.siteFeature.assign(site)
        else:
            self.siteFeatures[0].assign(site)

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
        #========================================================

    def build_pkconv_unet(self, masked, mask, training=True):
        # pdb.set_trace()
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


    def L1_site(self, mask):
        siteDelta = self.siteFeature - self.initValue
        _mask = mask[:1]

        def loss_L1site(y_true, y_pred):
            if self.obsOnlyL1:
                res = K.sum(K.abs(siteDelta*_mask))/K.sum(_mask)
            else:
                res = K.mean(siteDelta)
            return res

        return loss_L1site

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

            l1SiteLoss = self.L1_site(mask)(y_true, y_pred)

            # # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            res = w1*l1 + w2*l2 + l1SiteLoss
            # total_loss = tf.add(w1*l1,w2*l2,name="loss_total")
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
                    self.L1_site(mask),
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
        # sys.exit()
        
    def call(self, data, training=True):# data=(masked_imgs, mask)
        (masked_imgs, mask) = data
        output = self.build_pkconv_unet(masked_imgs, mask, training=training)
        return output

    # テスト用
    def setSiteFeature(self, site):
        self.siteFeature.assign(site)

    def getSiteFeature(self):
        return self.siteFeature.numpy()


#=====================================
class PConv_ConditionalSite(PConvLearnSite):
    def __init__(self, siteNum=3, opeType='mul', obsOnlyL1 = False, siteLayers=[1], localLasso=False, **kwargs):
        super().__init__(**kwargs)

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
        if self.localLasso:
            self.lasso_conv = functools.partial(
                nn_ops.convolution_v2,
                strides=[3,3],
                padding="SAME",
                dilations=[1,1],
                data_format="NHWC",
                name="lassoConv")
            self.lasso_kernel = np.ones([6,6,1,1]).astype("float32")

        # エンコーダの層数
        self.encLayerNum = 5
        # 位置特性を適用する層数(lnum)
        self.siteLayers = siteLayers
        # 位置特性の数
        self.siteNum = siteNum
        self.lnum = len(list(set(self.siteLayers)))
        
        self.initValues = []
        self.siteFeatures = []
        self.exist_imgs = []
        # 位置特性の初期値・学習するパラメータを設定
        for lind in self.siteLayers:
            i = lind - 1
            # 1＊縦＊横＊位置特性の数
            size = [1,self.img_rows//(2**i),self.img_cols//(2**i),siteNum]
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

    def build_pconv_unet(self, masked, mask, labels, training=True):
        # masked:[N,W,H,C]の欠損あり画像
        # labels:[N,class]のOneHotベクトル

        # 位置特性の適用
        def applySite(x,i):
            # その層に位置特性
            if i in self.siteLayers:
                # i:何層目か(1~5)
                # wind:ある層の位置特性の位置特性リスト(self.siteFeatures)でのインデックス
                wind = self.siteLayers.index(i)

                _s = self.siteFeatures[wind]
                _s = K.tile(_s,[x.shape[0],1,1,x.shape[-1]]) # _s:[N,w,h,class]
                pdb.set_trace()
                
                # lab:[N,w,h,class]
                lab = K.tile(labels[:,np.newaxis,np.newaxis,:],[1,_s.shape[1],_s.shape[2],1])
                _s = K.sum(_s*lab,axis=-1, keepdims=True)

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
        
        return outputs

    def call(self, data, training=True):# data=(masked_imgs, mask, labels)
        (masked_imgs, mask, labels) = data
        output = self.build_pconv_unet(masked_imgs, mask, labels, training=training)
        return output

    def encode_mask(self, mask):
        e_conv1, e_mask1 = self.encoder1(mask, mask, istraining=False)
        e_conv2, e_mask2 = self.encoder2(e_conv1, e_mask1, istraining=False)
        e_conv3, e_mask3 = self.encoder3(e_conv2, e_mask2, istraining=False)
        e_conv4, e_mask4 = self.encoder4(e_conv3, e_mask3, istraining=False)
        e_conv5, e_mask5 = self.encoder5(e_conv4, e_mask4, istraining=False)
        return [e_mask1,e_mask2,e_mask3,e_mask4,e_mask5]

    def LocalLasso_site(self, mask):

        siteDeltas = [self.siteFeatures[i] - self.initValues[i] for i,_ in enumerate(self.siteLayers)]
        siteDeltas = [d*d for d in siteDeltas] # 二乗誤差
        # エンコーダ各階層のマスク画像のリスト
        emasks = self.encode_mask(mask)
        masks = []

        for i,lind in enumerate(self.siteLayers):
            if lind==1:
                masks.append(mask)
            else:
                masks.append(emasks[lind-2][:,:,:,:1])

        def localLasso_loss(y_true, y_pred):
            res = 0
            for i,_ in enumerate(self.siteLayers):
                _site = siteDeltas[i]
                for j in range(self.siteNum):
                    _s = _site[0,:,:,j]
                    if self.obsOnlyL1:
                        # pdb.set_trace()
                        lasso_mask = self.lasso_conv(masks[i], self.lasso_kernel)
                        lasso_mask = K.clip(lasso_mask, 0, 1)
                        lasso_img = self.lasso_conv(_s*masks[i], self.lasso_kernel)
                        res = K.sum(K.abs(lasso_img*lasso_mask))/K.sum(lasso_mask)
                    else:
                        lasso_img = self.lasso_conv(_s, self.lasso_kernel)
                        res = K.mean(lasso_img)
            return res
        return localLasso_loss

    def L1_site(self, mask):
        # pdb.set_trace()
        siteDeltas = [self.siteFeatures[i] - self.initValues[i] for i,_ in enumerate(self.siteLayers)]
        # エンコーダ各階層のマスク画像のリスト
        # pdb.set_trace()
        emasks = self.encode_mask(mask)
        masks = []

        for i,lind in enumerate(self.siteLayers):
            if lind==1:
                masks.append(mask)
            else:
                masks.append(emasks[lind-2][:,:,:,:1])
        
        def loss_L1site(y_true, y_pred):
            res = 0
            for i,_ in enumerate(self.siteLayers):
                _site = siteDeltas[i]
                for j in range(self.siteNum):
                    _s = _site[0,:,:,j]
                    if self.obsOnlyL1:
                        res = K.sum(K.abs(_s*masks[i]))/K.sum(masks[i])
                    else:
                        res = K.mean(_s)
            return res
        return loss_L1site

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

            if self.localLasso:
                l1SiteLoss = self.LocalLasso_site(mask)(y_true, y_pred)
            else:
                l1SiteLoss = self.L1_site(mask)(y_true, y_pred)

            # # 各損失項の重み
            w1,w2,w3 = self.loss_weights
            res = w1*l1 + w2*l2 + l1SiteLoss
            # total_loss = tf.add(w1*l1,w2*l2,name="loss_total")
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
                    self.L1_site(mask),
                    self.LocalLasso_site(mask),
                    self.PSNR,
                ],
                run_eagerly=True
            )
   
    def plotSiteFeature(self, epoch=None, plotSitePath="",saveName=""):

        # pdb.set_trace()
        for i,sind in enumerate(self.siteLayers):
            
            _site = self.siteFeatures[i].numpy()

            for j in range(self.siteNum):
                _s = _site[0,:,:,j]    

                exist = self.exist_imgs[i]
                    
                plt.clf()
                plt.close()
                smax = np.max(_s)
                smin = np.min(_s)

                _s[exist==0] = -10000
                cmbwr = plt.get_cmap('viridis')
                cmbwr.set_under('black')

                plt.imshow(_s,cmap=cmbwr,norm=SqueezedNorm(vmin=smin,vmax=smax,mid=self.scenter))
                plt.colorbar(extend='both')
                if saveName == "":
                    saveName = f"siteFeature{epoch}"
                    
                plt.savefig(f"{plotSitePath}{os.sep}{saveName}_layer{sind}channel{j+1}.png")
    
    # テスト用
    def setSiteFeature(self, site):
        if self.onlyInputSite:
            self.siteFeature.assign(site)
        else:
            self.siteFeatures[0].assign(site)

    def getSiteFeature(self):
        return [s.numpy() for s in self.siteFeatures]


#=====================================
class branchPConv_lSite(PConvLearnSite):
    def __init__(self, **kwargs):
        super(branchPConv_lSite, self).__init__(**kwargs)
        self.phase = 0
        # 位置特性のConvolution
        self.siteConv1 = siteConv(filters=64, strides=(1,1), kernel_size=(3,3), padding='same', use_bias=False)
        self.siteConv2 = siteConv(filters=128, strides=(1,1), kernel_size=(3,3), padding='same', use_bias=False)

        # self.siteConv1 = Conv2D(filters=64, strides=(1,1), kernel_size=(3,3), padding='same', use_bias=False, name="site_conv")
        # self.siteConv2 = Conv2D(filters=128,strides=(1,1), kernel_size=(3,3), padding='same', use_bias=False, name="site_conv_1")

    def build_pconv_unet(self, masked, mask, training=True, extract="", phase=0):
        # 位置特性の適用
        def applySite(x,i,extract=""):
            # その層に位置特性
            if i in self.siteLayers:
                wind = self.siteLayers.index(i)
                _s = self.siteFeature if self.onlyInputSite else self.siteFeatures[wind]
                _s = K.tile(_s,[x.shape[0],1,1,1])
                # pdb.set_trace()
                # 位置特性のCNN
                sconv1 = self.siteConv1(_s)
                sconv2 = self.siteConv2(sconv1)

                if self.opeType=="add":
                    res = x + sconv2
                elif self.opeType=="mul":
                    res = x * sconv2
            else:
                res = x

            if extract != "":
                exec(f"res = {extract}")

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

        if extract != "":
            exec(f"outputs = {extract}")
                
        return outputs

    def plotSiteFeature(self, epoch=None, plotSitePath="", saveName="", plotfmap=True):

        super(branchPConv_lSite, self).plotSiteFeature(
            epoch=epoch,plotSitePath=plotSitePath,saveName=saveName
        )

        if plotfmap:

            for i in self.siteLayers:
                wind = self.siteLayers.index(i)
                _s = self.siteFeature if self.onlyInputSite else self.siteFeatures[wind]

                plt.clf()       
                plt.close()
                plot_chmax = 64
                plot_w, plot_h = [8,8]

                # 位置特性のCNN
                sconv1 = self.siteConv1(_s)
                sconv2 = self.siteConv2(sconv1)
                plotsconv2 = sconv2[0,:,:,:plot_chmax]
                
                smax = np.max(plotsconv2)
                smin = np.min(plotsconv2)

                # cmbwr = plt.get_cmap('viridis')
                cmbwr = plt.get_cmap('bwr')
                cmbwr.set_under('black')

                _, axes = plt.subplots(plot_h, plot_w, tight_layout=True, figsize=(plot_w*5, plot_h*5))
                for w in range(plot_w):
                    for h in range(plot_h):
                        axes[h,w].imshow(
                            plotsconv2[:,:, w + h*plot_w],
                            cmap=cmbwr,
                            norm=SqueezedNorm(vmin=smin,vmax=smax,mid=self.scenter),
                            interpolation='none'
                        )
                        axes[h,w].set_title(f"ch {w + h*plot_w}")
                
                if saveName == "":
                    saveName = f"siteFeature{epoch}"
                plt.savefig(f"{plotSitePath}{os.sep}{saveName}_sconv2.png")

    def compile(self, lr, mask, phase=0):
        # pdb.set_trace()
        met = [
            self.loss_total(mask),
            self.holeLoss(mask),
            self.validLoss(mask),
            self.tvLoss(mask),
            self.L1_site(mask),
            self.PSNR,
        ]
        if self.localLasso:
            met.append(self.LocalLasso_site(mask))
            
        if phase==0: # モデルを別で学習しない
            # InpaintingModelの親でコンパイル
            super(InpaintingModel, self).compile(
                    optimizer = Adam(lr=lr),
                    loss= self.loss_total(mask),
                    metrics=met,
                    run_eagerly=True
                )
        elif phase==1: # InpaintのPConvUNetのみ学習
            super(InpaintingModel, self).compile(
                optimizer = Adam(lr=lr),
                loss= self.loss_total(mask),
                metrics=met,
                run_eagerly=True
            )
        elif phase==2: # siteCNNのみ学習
            super(InpaintingModel, self).compile(
                optimizer = Adam(lr=lr),
                loss= self.loss_total(mask),
                metrics=met,
                run_eagerly=True
            )
   
    def changeTrainPhase(self, phase):

        self.phase = phase
        def setPconvTrainable(setBool):
            self.encoder1.trainable = setBool
            self.encoder2.trainable = setBool
            self.encoder3.trainable = setBool
            self.encoder4.trainable = setBool
            self.encoder5.trainable = setBool

            self.decoder6.trainable = setBool
            self.decoder7.trainable = setBool
            self.decoder8.trainable = setBool
            self.decoder9.trainable = setBool
            self.decoder10.trainable = setBool

        if phase==1:
            # pdb.set_trace()
            self.siteConv1.trainable = False
            self.siteConv2.trainable = False
            setPconvTrainable(True)
        elif phase==2:
            self.siteConv1.trainable = True
            self.siteConv2.trainable = True
            setPconvTrainable(False)

    def siteFreeze(self):
        for _s in self.siteFeatures:
            _s.trainable = False        

    def train_step(self, data):# data=(masked_imgs, mask, gt_imgs)
        (masked_imgs, mask, gt_imgs) = data[0]
        
        # 勾配の取得
        with tf.GradientTape() as tape:
            pred_imgs = self((masked_imgs, mask), training=True)
            loss = self.compiled_loss(gt_imgs, pred_imgs)

        # 勾配によるパラメータの更新
        trainable_vars = self.trainable_variables
        # pdb.set_trace()

        if self.phase == 1: # 位置特性の学習なし
            trainable_vars = [
                v for v in trainable_vars if ('siteFeature' not in  v.name) and ('site_conv' not in v.name)
            ]

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 評価値の更新
        self.compiled_metrics.update_state(gt_imgs, pred_imgs)

        return {m.name: m.result() for m in self.metrics}

        



class sharePConv_lSite(PConvLearnSite):
    def build_pconv_unet(self, masked, mask, training=True, extract=""):
        # 位置特性の適用
        def applySite(x,i,extract=""):
            # その層に位置特性
            if i in self.siteLayers:
                wind = self.siteLayers.index(i)
                _s = self.siteFeature if self.onlyInputSite else self.siteFeatures[wind]
                _s = K.tile(_s,[x.shape[0],1,1,1])

                # 位置特性のCNN
                sconv1,smask1 = self.encoder1(_s, mask, istraining=False)
                sconv2,_ = self.encoder2(sconv1, smask1, istraining=False)

                if self.opeType=="add":
                    res = x + sconv2
                elif self.opeType=="mul":
                    res = x * sconv2
            else:
                res = x

            if extract != "":
                exec(f"res = {extract}")

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

        if extract != "":
            exec(f"outputs = {extract}")
                
        return outputs

        


