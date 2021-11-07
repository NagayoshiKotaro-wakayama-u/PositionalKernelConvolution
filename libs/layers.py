import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec, Concatenate, BatchNormalization, Activation, LeakyReLU, UpSampling2D
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn, nn_ops, array_ops
import six
import functools
import pdb

#=================================================================
# pconv
class PConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(PConv2D, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(min_ndim=4),InputSpec(min_ndim=4)]

    def build(self,input_shape):
        # pdb.set_trace()
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

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None

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

    def call(self, img_in, mask_in, istraining=True):
        conv,mask = self.pconv((img_in,mask_in))
        if self.bn:
            conv = self.batchnorm(conv,training=istraining)
        conv = self.relu(conv)
        return (conv, mask)


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

#=================================================================
# pkconv
class PKConv(tf.keras.layers.Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, posEmbChan=1,use_site=False,use_sCNN=False,opeType="add",eachChannel=False,
        sConvKernelLearn=False,site_range=[0,1],sklSigmoid=False,sConvChan=None,learnMultiSiteW=False, **kwargs):
        super().__init__(*args, **kwargs)
        # pdb.set_trace()

        if use_site or use_sCNN:
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4), InputSpec(ndim=4)]
        else:
            self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]
        
        self.posEmbChan= posEmbChan
        self.use_site = use_site
        self.use_sCNN = use_sCNN
        self.opeType = opeType
        self.eachChannel = eachChannel
        self.sConvKernelLearn = sConvKernelLearn
        self.sConvChan = [posEmbChan]*2 if sConvChan is None else sConvChan
        if use_site:
            self.site_min = site_range[0]
            self.site_max = site_range[1]
        
        self.multiSiteW_Learn = learnMultiSiteW

        # 学習するときのみ代入
        self.sklSigmoid = sklSigmoid if self.sConvKernelLearn else False

    def build(self, input_shape):        
        """Adapted from original _Conv() layer of Keras        
        param input_shape: list of dimensions for [img, mask]
        """
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        
        self.input_dim = input_shape[0][channel_axis]

        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        
        # 足し算か掛け算か
        if self.opeType == "add":
            self.onesKernel = K.constant(np.ones(kernel_shape))
        elif self.opeType == "mul":
            self.bias_initializer = keras.initializers.Ones()

        # Mask kernel
        self.kernel_mask = K.ones(shape=kernel_shape)

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        # 位置特性を使う場合
        if self.use_site:
            self.bias = None

            # 位置特性のカーネル（学習可能か固定か）
            if self.sConvKernelLearn:
                self.kernel_site = K.ones(shape=self.kernel_size + (self.sConvChan[0], self.sConvChan[1]))/np.prod(self.kernel_size)
            else:
                self.kernel_site = K.constant(
                    np.ones(self.kernel_size + (self.sConvChan[0], self.sConvChan[1]))
                )
                
            # 複数の位置特性の加重和を計算するための学習可能な重み
            if self.multiSiteW_Learn:
                # pdb.set_trace()
                self.multiSiteW = self.add_weight(
                    shape=self.posEmbChan,initializer=keras.initializers.Ones(),
                    name='multiSiteW',trainable=True
                )
            
        elif self.use_sCNN:
            self.bias = None
        else:
            if self.eachChannel:
                bias_shape = (1, input_shape[0][1].value, input_shape[0][2].value, self.input_dim)
            else:
                bias_shape = (1, input_shape[0][1].value, input_shape[0][2].value, self.posEmbChan)
            self.bias = self.add_weight(shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        # 特徴マップのチャネル数（位置特性のチャネル数を特徴マップと合わせるために使用）
        self.tileNum = input_shape[0][3].value
        self.built = True
    
    def call(self, inputs, mask=None):
        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) > 3 or len(inputs) <= 1:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [img, mask] or [img, mask, site]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)
        # pdb.set_trace()
        # 位置特性を使う場合
        if len(inputs)==3:
            bias = K.spatial_2d_padding(inputs[2], self.pconv_padding, self.data_format)
        
            # 位置特性を次の層に渡す際のConvolution
            # kernel_site はnp.ones()の定数か変数
            if self.use_site:
                # 次の層へ位置特性を伝搬
                site_output = K.conv2d(bias, self.kernel_site, strides=self.strides, padding='valid',
                    data_format=self.data_format, dilation_rate=self.dilation_rate)

                # 次層に渡す前にSigmoidにかける
                if self.sklSigmoid:
                    site_output = K.sigmoid(site_output) # 0~1
                    d_site = self.site_max-self.site_min 
                    site_output = site_output*d_site + self.site_min
                else:
                    site_output = K.clip(site_output, self.site_min, self.site_max)

                # 元の位置特性のチャネル数（枚数）が一枚の時に、
                # 前の層から渡されたbias(位置特性)が複数チャネルだった場合
                if bias.shape[3] > 1 and self.posEmbChan == 1:
                    # 位置エンコーディングに使うbiasは1x1Convで1チャネルにする
                    bias = K.conv2d(bias, K.constant(np.ones([1,1,bias.shape[3],1])), strides=(1,1), padding='valid',
                        data_format=self.data_format, dilation_rate=self.dilation_rate)
                
        else:
            bias = K.spatial_2d_padding(self.bias, self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(masks, self.kernel_mask, strides=self.strides, padding='valid',
            data_format=self.data_format, dilation_rate=self.dilation_rate)

        if self.opeType == "add": # 位置特性を足し算(W+P)X
            # Apply convolutions to image (W*X)
            wx = K.conv2d(
                (images*masks), self.kernel, 
                strides=self.strides,
                padding='valid',
                data_format=self.data_format,
                dilation_rate=self.dilation_rate
            )

            # 全チャネルに対して位置特性を用意している場合
            if self.eachChannel:
                # pdb.set_trace()
                px = bias*images
                px = K.conv2d(
                    px, self.onesKernel, 
                    strides=self.strides,
                    padding='valid',
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate
                )

            # 特徴マップのチャネル数より位置特性の枚数が少ない場合
            else:
                # 末尾に次元を追加してタイル（位置特性ごとに塊ができるようにタイルするため）
                posEmb = K.tile(tf.expand_dims(bias[:1],-1),[1,1,1,1,self.tileNum])
                posEmb = tf.reshape(posEmb,shape=[1]+images.shape[1:3]+[self.tileNum*self.posEmbChan])
                tiled_images = K.tile(images,[1,1,1,self.posEmbChan])

                # PX = P(位置特性)  ×  X(特徴マップ)
                pxs = posEmb*tiled_images

                px = 0
                for c_ind in range(self.posEmbChan):
                    pxi = pxs[:,:,:,self.tileNum*c_ind:self.tileNum*(c_ind+1)]
                    pxi = K.conv2d(
                        pxi, self.onesKernel, #マスク処理いるのでは？
                        strides=self.strides,
                        padding='valid',
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate
                    )
                    px = pxi + px # TODO:いずれはAttenntionなどによる重みつき和を計算したい

            img_output = wx + px

        elif self.opeType == "mul": # 位置特性を掛け算
            # pdb.set_trace()
            tiled_images = K.tile(images,[1,1,1,self.posEmbChan])
            posEmb = K.tile(bias,[1,1,1,self.tileNum])
            pxs = posEmb*tiled_images
            
            img_output = 0
            if self.posEmbChan > 1 and self.multiSiteW_Learn:
                multiSiteW_softed = K.softmax(self.multiSiteW)
            for c_ind in range(self.posEmbChan):
                pxi = pxs[:,:,:,self.tileNum*c_ind:self.tileNum*(c_ind+1)]
                img_out = K.conv2d(
                    (pxi*masks), self.kernel, 
                    strides=self.strides,
                    padding='valid',
                    data_format=self.data_format,
                    dilation_rate=self.dilation_rate
                )
                
                if self.posEmbChan > 1 and self.multiSiteW_Learn:# 重み和の計算で重みを学習するかどうか
                    img_output += multiSiteW_softed[c_ind] * img_out
                else: # 平均を取る
                    img_output += 1/self.posEmbChan * img_out

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)
        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)
        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output
        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)

        if self.use_site:
            outputs = [img_output, mask_output, site_output]
        else:
            outputs = [img_output, mask_output]

        return outputs
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)

            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]

        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]


class PKEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, iterNum, posEmbChan=1, use_site=False, use_sCNN=False, opeType="add", bn=True, istraining=True,
        eachChannel=False,sConvKernelLearn=False,sConvChan=None,site_range=[0,1],sklSigmoid=False,learnMultiSiteW=False):
        
        super().__init__()
        
        self.pkconv = PKConv(filters, kernel_size, posEmbChan=posEmbChan, use_site=use_site, use_sCNN=use_sCNN,
            opeType=opeType, eachChannel=eachChannel,sConvKernelLearn=sConvKernelLearn,sConvChan=sConvChan,
            site_range=site_range,sklSigmoid=sklSigmoid,learnMultiSiteW=learnMultiSiteW, strides=2, padding='same')
        
        self.count = iterNum
        self.bn = bn
        self.training = istraining
        self.batchnorm = BatchNormalization(name='EncBN'+str(self.count))
        self.relu = Activation('relu')
        self.opeType = opeType
        self.use_sCNN = use_sCNN

    def call(self,img_in,mask_in,site_in=None):
        if site_in==None:
            inputs = [img_in,mask_in]
        elif self.use_sCNN:
            inputs=[img_in,mask_in,site_in]
        else:
            inputs=[img_in,mask_in,site_in]

        output = self.pkconv(inputs)
        conv = output[0]

        if self.bn:
            conv = self.batchnorm(conv,training=self.training)
        conv = self.relu(conv)
        output[0] = conv
        
        return output

#=================================================================
# siteconv
class siteConv(tf.keras.layers.Conv2D):
    def __init__(self, *args):
        pass

    def build(self, input_shapes):
        pass

    def call(self, inputs):
        pass
