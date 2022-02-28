import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Concatenate, BatchNormalization, Activation, LeakyReLU, UpSampling2D, MaxPool2D
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn, nn_ops, array_ops
from tensorflow.keras.constraints import NonNeg
import six
import functools
import pdb
import numpy as np
#=================================================================
# pconv
class PConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(PConv2D, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(min_ndim=4),InputSpec(min_ndim=4)]

    def build(self,input_shape):
        # pdb.set_trace()
        # 入力のShape
        input_shape = tensor_shape.TensorShape(input_shape[0])
        input_channel = self._get_input_channel(input_shape)
        # カーネルのShape
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=self.trainable,
            dtype=self.dtype)

        self.kernel_mask = tf.keras.backend.ones(kernel_shape)

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=self.trainable,
                dtype=self.dtype)
        else:
            self.bias = None

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper() # SAME
        else:
            tf_padding = self.padding

        tf_dilations = list(self.dilation_rate) # [1,1]
        tf_strides = list(self.strides) # [2,2]

        tf_op_name = self.__class__.__name__ # PConv2D

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)

        self.built = True

    def call(self, inputs):
        _img, _mask = inputs
        outputs_img = self._convolution_op(_img*_mask, self.kernel)
        outputs_mask = self._convolution_op(_mask, self.kernel_mask)
        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (outputs_mask + 1e-8)
        # Clip output to be between 0 and 1
        outputs_mask = K.clip(outputs_mask, 0, 1)
        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * outputs_mask
        # Normalize iamge output
        outputs_img = outputs_img * mask_ratio

        if self.use_bias:
            output_rank = outputs_img.shape.rank
            # Handle multiple batch dimensions.
            if output_rank is not None and output_rank > 2 + self.rank:
                def _apply_fn(o):
                    return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                outputs_img = nn_ops.squeeze_batch_dims(
                    outputs_img, _apply_fn, inner_rank=self.rank + 1)
            else:
                outputs_img = nn.bias_add(
                    outputs_img, self.bias, data_format=self._tf_data_format)

        if self.activation is not None:
            outputs_img = self.activation(outputs_img)

        return (outputs_img, outputs_mask)

# encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, encoderNumber, strides=(2,2), bn=True, use_bias=False):
        super(Encoder, self).__init__()
        if isinstance(kernel_size,int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size,tuple):
            raise ValueError("kernel_size must be tuple of int or scalar of int")

        self.pconv = PConv2D(filters=filters, strides=strides, kernel_size=kernel_size, padding='same', use_bias=use_bias)
        self.bn = bn
        self.encoderNumber = encoderNumber
        self.batchnorm = BatchNormalization(name=f'EncBN_{self.encoderNumber}')
        self.relu = Activation('relu')
        self.use_bias = use_bias
        self.strides = strides

        if strides == (1,1):
            # pdb.set_trace()
            self.pooling = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

    def call(self, img_in, mask_in, istraining=True):
        # pdb.set_trace()
        conv,mask = self.pconv((img_in,mask_in))
        if self.bn:
            conv = self.batchnorm(conv,training=istraining)
        conv = self.relu(conv)

        if self.strides==(1,1):
            conv = self.pooling(conv)
            mask = self.pooling(mask)
        
        return (conv, mask)

    def switchTrainable(self,setBool=False):
        self.trainable = setBool

# decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, bn=True, use_bias=False):
        super(Decoder, self).__init__()
        if isinstance(kernel_size,int):# intならタプルに
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size,tuple):
            raise ValueError("kernel_size must be tuple of int or scalar of int")

        self.bn = bn
        self.upsample = UpSampling2D(size=(2,2))
        self.concatenate = Concatenate(axis=3)
        self.pconv = PConv2D(filters=filters, strides=(1,1), kernel_size=kernel_size, padding='same', use_bias=use_bias)
        self.batchnorm = BatchNormalization()
        self.leakyrelu = LeakyReLU(alpha=0.2)
        self.use_bias = use_bias
    
    def call(self,img_in,mask_in, e_conv, e_mask, istraining=True):
        up_img = self.upsample(img_in)
        up_mask = self.upsample(mask_in)
        conc_img = self.concatenate([e_conv,up_img])
        conc_mask = self.concatenate([e_mask,up_mask])
        conv,mask = self.pconv([conc_img,conc_mask])
        if self.bn:
            conv = self.batchnorm(conv,training=istraining)
        conv = self.leakyrelu(conv)
        return conv, mask

    def switchTrainable(self,setBool=False):
        self.trainable = setBool
        pdb.set_trace()

#=================================================================
# pkconv
class PKConv(tf.keras.layers.Conv2D):
    def __init__(self, *args, siteInputChan=1, opeType="mul",
        sConvKernelLearn=False, learnMultiSiteW=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(min_ndim=4),InputSpec(min_ndim=4),InputSpec(min_ndim=4)]

        self.siteInputChan= siteInputChan
        self.opeType = opeType
        self.sConvKernelLearn = sConvKernelLearn
        self.multiSiteW_Learn = learnMultiSiteW

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape[0])
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        self.kernel_mask = tf.keras.backend.ones(kernel_shape)

        # 足し算か掛け算か
        if self.opeType == "add":
            self.onesKernel = K.constant(np.ones(kernel_shape))
        elif self.opeType == "mul":
            self.bias_initializer = tf.keras.initializers.Ones()

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        # 特徴マップのチャネル数//位置特性のチャネル数
        #（位置特性のチャネル数を特徴マップと合わせるために使用）
        # self.tileNum = input_shape[3]
        self.tileNum = 1

        if self.siteInputChan > 1 and self.multiSiteW_Learn:
            self.multiSiteW = self.add_weight(
                name='multiSiteW',
                shape=[self.siteInputChan],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True,
                dtype=self.dtype)


        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        
        # Convolution Operation
        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)

        self.built = True


    def call(self, inputs, mask=None):
        _img, _mask, _site = inputs
        if _site.shape[0] == 1:
            _site = K.tile(_site,[_img.shape[0],1,1,1])

        if self.opeType == "add": # 位置特性を足し算(W+P)X
            # Apply convolutions to image (W*X)
            wx = self._convolution_op(_img*_mask, self.kernel)

            # 特徴マップのチャネル数より位置特性の枚数が少ない場合 ← 考慮しないことに(2022/02/07)
            # 末尾に次元を追加してタイル（位置特性ごとに塊ができるようにタイルするため）
            # posEmb = K.tile(tf.expand_dims(_site[:1],-1),[1,1,1,1,self.tileNum])
            # posEmb = tf.reshape(posEmb,shape=[1]+_img.shape[1:3]+[self.tileNum*self.siteInputChan])
            # tiled_images = K.tile(_img,[1,1,1,self.siteInputChan])
            
            posEmb = _site

            ## PXにもマスクをかける ← 2022/02/28
            # PX = P(位置特性)  ×  X(特徴マップ) x mask
            pxs = posEmb*_img*_mask

            # 複数チャネル＆加重和のための重み
            if self.siteInputChan > 1 and self.multiSiteW_Learn:
                multiSiteW_softed = K.softmax(self.multiSiteW)

            # 各チャネルの統合
            # pdb.set_trace()
            px = self._convolution_op(pxs,self.onesKernel)
            # px = 0
            # for c_ind in range(self.siteInputChan):
            #     pxi = pxs[:,:,:,self.tileNum*c_ind:self.tileNum*(c_ind+1)]
            #     pxi = self._convolution_op(pxi, self.onesKernel)
            #     # px = pxi + px # TODO:いずれはAttenntionなどによる重みつき和を計算したい
            #     if self.siteInputChan > 1 and self.multiSiteW_Learn:# 重み和の計算で重みを学習するかどうか
            #         px += multiSiteW_softed[c_ind] * pxi
            #     else: # 平均を取る
            #         px += pxi / self.siteInputChan

            outputs_img = wx + px

        elif self.opeType == "mul": # 位置特性を掛け算
            # pdb.set_trace()
            tiled_images = K.tile(_img,[1,1,1,self.siteInputChan]) # 入力画像・特徴マップ
            # posEmb = K.tile(_site,[1,1,1,self.tileNum]) # 位置特性
            posEmb = tf.reshape(tf.tile(tf.expand_dims(_site,-1),[1,1,1,1,self.tileNum]),tiled_images.shape)
            pxs = posEmb*tiled_images
            
            outputs_img = 0

            # 複数チャネル＆加重和のための重み
            if self.siteInputChan > 1 and self.multiSiteW_Learn:
                multiSiteW_softed = K.softmax(self.multiSiteW)
                
            for c_ind in range(self.siteInputChan):
                pxi = pxs[:,:,:,self.tileNum*c_ind:self.tileNum*(c_ind+1)]
                img_out = self._convolution_op(pxi*_mask, self.kernel)
                
                if self.siteInputChan > 1 and self.multiSiteW_Learn:# 重み和の計算で重みを学習するかどうか
                    outputs_img += multiSiteW_softed[c_ind] * img_out
                else: # 平均を取る
                    outputs_img += 1/self.siteInputChan * img_out

        outputs_mask = self._convolution_op(_mask, self.kernel_mask)
        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (outputs_mask + 1e-8)
        # Clip output to be between 0 and 1
        outputs_mask = K.clip(outputs_mask, 0, 1)
        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * outputs_mask
        # Normalize iamge output
        outputs_img = outputs_img * mask_ratio

        # Apply activations on the image
        if self.activation is not None:
            outputs_img = self.activation(outputs_img)

        outputs = [outputs_img, outputs_mask]

        return outputs

class PKEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, siteInputChan=1, opeType="mul",
        sConvKernelLearn=False, site_range=[0.1,1], learnMultiSiteW=False, siteSigmoid=False,
        siteClip=False, bn=True):
        
        super().__init__()
        
        self.pkconv = PKConv(filters, kernel_size, siteInputChan=siteInputChan, 
            opeType=opeType, sConvKernelLearn=sConvKernelLearn, 
            learnMultiSiteW=learnMultiSiteW, strides=2, padding='same')
        
        self.bn = bn
        self.batchnorm = BatchNormalization(name='EncBN')
        self.relu = Activation('relu')
        self.opeType = opeType

        self.siteSigmoid = siteSigmoid
        self.siteClip = siteClip
        if siteSigmoid or siteClip:
            self.sitemin = min(site_range)
            self.sitemax = max(site_range)

    def call(self,img_in,mask_in,site_in, istraining=True):
        # 位置特性を適用する前に値域を制限
        if self.siteSigmoid:
            site = K.sigmoid(site_in)
            site = site * (self.sitemax - self.sitemin) + self.sitemin
        elif self.siteClip:
            site = K.clip(site_in, self.sitemax, self.sitemin)
        else:
            site = site_in

        output = self.pkconv((img_in, mask_in, site))
        conv = output[0] # img

        if self.bn:
            conv = self.batchnorm(conv,training=istraining)
        conv = self.relu(conv)
        output[0] = conv # img
        
        return output

#=================================================================
# siteconv
class siteConv(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        # pdb.set_trace()
        super(siteConv, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(min_ndim=4)]
        # if nonNegative:
        #     self.kernel_constraint = NonNeg()

    # def build(self, input_shape):
    #     # pdb.set_trace()
    #     input_shape = tensor_shape.TensorShape(input_shape)
    #     input_channel = self._get_input_channel(input_shape)
    #     kernel_shape = self.kernel_size + (input_channel, self.filters)
    #     self.kernel = self.add_weight(
    #         name='kernel',
    #         shape=kernel_shape,
    #         initializer=self.kernel_initializer,
    #         regularizer=self.kernel_regularizer,
    #         constraint=self.kernel_constraint,
    #         trainable=self.trainable,
    #         dtype=self.dtype)
        
    #     # Convert Keras formats to TF native formats.
    #     if self.padding == 'causal':
    #         tf_padding = 'VALID'  # Causal padding handled in `call`.
    #     elif isinstance(self.padding, six.string_types):
    #         tf_padding = self.padding.upper()
    #     else:
    #         tf_padding = self.padding

    #     tf_dilations = list(self.dilation_rate)
    #     tf_strides = list(self.strides)
    #     tf_op_name = self.__class__.__name__

    #     self._convolution_op = functools.partial(
    #         nn_ops.convolution_v2,
    #         strides=tf_strides,
    #         padding=tf_padding,
    #         dilations=tf_dilations,
    #         data_format=self._tf_data_format,
    #         name=tf_op_name)

    #     self.built = True

    def call(self, inputs):
        _site = inputs
        outputs_img = self._convolution_op(_site, self.kernel)
        if self.activation is not None:
            outputs_img = self.activation(outputs_img)
        return outputs_img


class siteDeconv(siteConv):
    def __init__(self, *args, **kwargs):
        super(siteDeconv, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        super(siteDeconv, self).build(input_shape)
        self.upsample = UpSampling2D(size=(2,2))

    def call(self, inputs):
        _site = inputs
        upsampled = self.upsample(_site)
        outputs_img = self._convolution_op(upsampled, self.kernel)
        if self.activation is not None:
            outputs_img = self.activation(outputs_img)
        return outputs_img

#=================================================================
# multimodal transfer
class MMCM(tf.keras.layers.Layer):
    def __init__(self, c):
        super().__init__()
    

    def call(self, A, B):
        pass


