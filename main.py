import tensorflow as tf
import matplotlib.pylab as plt
import os
import pickle
import numpy as np
import pdb
import copy
from libs.models import PConvUnet,PConvUnetModel
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Training script for PConv inpainting')
    parser.add_argument('experiment',type=str,help='name of experiment, e.g. \'normal_PConv\'')
    parser.add_argument('-dataset','--dataset',type=str,default='gaussianToyData',help='name of dataset directory (default=gaussianToyData)')
    parser.add_argument('-epochs','--epochs',type=int,default=400,help='training epoch')
    parser.add_argument('-imgw','--imgw',type=int,default=512,help='input width')
    parser.add_argument('-imgh','--imgh',type=int,default=512,help='input height')
    parser.add_argument('-lr','--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('-lossw','--lossWeights',type=lambda x:list(map(float,x.split(","))),default="1,6,0.1")
    return  parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    tf.config.run_functions_eagerly(True)

    # 実験に用いるディレクトリを作成
    experiment_path = ".{0}experiment{0}{1}_logs".format(os.sep,args.experiment)
    loss_path = f"{experiment_path}{os.sep}losses"
    log_path = f"{experiment_path}{os.sep}logs"
    test_path = f"{experiment_path}{os.sep}test_samples"
    # site_path = f"data{os.sep}new_siteImages{os.sep}"

    for DIR in [experiment_path,loss_path,log_path,test_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    epochs = args.epochs
    dataset = args.dataset # データセットのディレクトリ
    dspath = ".{0}data{0}{1}{0}".format(os.sep,dataset)

    # 各pickleデータのパス
    TRAIN_PICKLE = dspath+"train.pickle"
    TRAIN_MASK_PICKLE = dspath+"train_mask.pickle"
    VALID_PICKLE = dspath+"valid.pickle"
    VALID_MASK_PICKLE = dspath+"valid_mask.pickle"
    TEST_PICKLE = dspath+"test.pickle"
    TEST_MASK_PICKLE = dspath+"test_mask.pickle"

    train_Num = pickle.load(open(TRAIN_PICKLE,"rb"))["images"].shape[0] # 画像の枚数をカウント
    valid_Num = pickle.load(open(VALID_PICKLE,"rb"))["images"].shape[0]
    img_w = args.imgw
    img_h = args.imgh
    shape = (img_h, img_w)

    # バッチサイズはメモリサイズに合わせて調整が必要
    batchsize = 5 # バッチサイズ
    steps_per_epoch = train_Num//batchsize # 1エポック内のiteration数

    # generatorを作成
    trainImg = pickle.load(open(TRAIN_PICKLE,"rb"))["images"]
    trainMask = pickle.load(open(TRAIN_MASK_PICKLE,"rb"))
    trainMasked = trainImg*trainMask

    validImg = pickle.load(open(VALID_PICKLE,"rb"))["images"]
    validMask = pickle.load(open(VALID_MASK_PICKLE,"rb"))
    validMasked = validImg*validMask

    #---------------------------------------------------------------
    # Build the model
    ## keyargs
    # keyArgs = {"img_rows":img_h,"img_cols":img_w,"lr":args.lr,"loss_weights":args.lossWeights}
    # pconv_unet = PConvUnet(**keyArgs)
    pconv_unet = PConvUnetModel()
    # pdb.set_trace()
    pconv_unet.compile(args.lr, trainMask[0:1])

    checkpoint_path = "autoencoder_training/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)]
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_PSNR', min_delta=0.1, patience=10,
            verbose=1, mode='max', baseline=None, restore_best_weights=True
        )
    )

    # model = traceableModel()
    history = pconv_unet.fit(
        (trainMasked, trainMask, trainImg),
        batch_size=5,
        validation_data=((validMasked, validMask), validImg),
        epochs=args.epochs,
        validation_split=0.1, 
        callbacks=callbacks)
