import tensorflow as tf
import matplotlib.pylab as plt
import os
import pickle
import numpy as np
import pdb
import copy
from libs.util import loadSiteImage,plotDatas
from argparse import ArgumentParser
from libs.modelConfig import parse_args,modelBuild
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


if __name__ == "__main__":
    args = parse_args(isTrain=True)
    tf.config.run_functions_eagerly(True)

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    # フェーズ
    ph = ""
    if args.phase > 1:
        ph = f"{args.phase}"

    # 実験に用いるディレクトリを作成
    experiment_path = ".{0}experiment{0}{1}_logs".format(os.sep,args.experiment)
    loss_path = f"{experiment_path}{os.sep}losses{ph}"
    log_path = f"{experiment_path}{os.sep}logs{ph}"
    test_path = f"{experiment_path}{os.sep}test_samples{ph}"
    site_path = f"data{os.sep}siteImage{os.sep}"

    for DIR in [experiment_path,loss_path,log_path,test_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)

    epochs = args.epochs
    dataset = args.dataset # データセットのディレクトリ
    dspath = ".{0}data{0}{1}{0}".format(os.sep,dataset)
    existPointPath = f"data{os.sep}sea.png" if "quake" in dataset else ""

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

    # データのロード
    trainImg = pickle.load(open(TRAIN_PICKLE,"rb"))["images"]
    trainMask = pickle.load(open(TRAIN_MASK_PICKLE,"rb"))
    trainMasked = trainImg*trainMask

    validImg = pickle.load(open(VALID_PICKLE,"rb"))["images"]
    validMask = pickle.load(open(VALID_MASK_PICKLE,"rb"))
    validMasked = validImg*validMask

    testImg = pickle.load(open(TEST_PICKLE,"rb"))["images"]
    testMask = pickle.load(open(TEST_MASK_PICKLE,"rb"))
    testMasked = testImg*testMask
    #---------------------------------------------------------------
    # Build the model
    if args.PKConv:# カーネルに対して位置特性を導入
        # モデルのビルド
        model = modelBuild("pkconvAndSiteCNN",args)

        # 位置特性の読み込み(loadSiteImage内で正規化)
        siteImg = loadSiteImage(
            [site_path + s for s in args.sitePath],
            smax=args.siteMax,
            smin=args.siteMin
        )
        tra_siteImgs = np.tile(siteImg,(trainImg.shape[0],1,1,1))
        val_siteImgs = np.tile(siteImg,(validImg.shape[0],1,1,1))
        
        # データセット
        trainData = (trainMasked, trainMask, tra_siteImgs, trainImg)
        validData = ((validMasked, validMask, val_siteImgs), validImg)
        test_sample = (testMasked[0:1], testMask[0:1], siteImg) 
    
    elif args.concatPConv:# 位置特性をチャネル方向に結合
        # モデルのビルド
        model = modelBuild("conc-pconv",args)
        
        # 位置特性の読み込み(loadSiteImage内で正規化)
        siteImg = loadSiteImage(
            [site_path + s for s in args.sitePath],
            smax=args.siteMax,
            smin=args.siteMin
        )
        tra_siteImgs = np.tile(siteImg,(trainImg.shape[0],1,1,1))
        val_siteImgs = np.tile(siteImg,(validImg.shape[0],1,1,1))
        # 入力に位置特性をConcat
        trainMasked = np.concatenate([trainMasked,tra_siteImgs],axis=3)
        validMasked = np.concatenate([validMasked,val_siteImgs],axis=3)

        # データセット
        trainData = (trainMasked, trainMask, trainImg)
        validData = ((validMasked, validMask), validImg)
        test_sample = (
            np.concatenate([testMasked[0:1],val_siteImgs[0:1]],axis=3),
            testMask[0:1]
        )

    elif args.cPConv: # 震源別・ラベル情報別に位置特性を学習し使い分ける
        trainLab = pickle.load(open(TRAIN_PICKLE,"rb"))["labels"]
        #TODO:ラベル情報をOnehotベクトルに変換
        trainData = (trainMasked, trainMask, trainImg)
        model = modelBuild("conditionalPConv",args)

    else:
        # 学習可能な位置特性を持つPConvUNet
        if args.learnSitePConv:
            model = modelBuild("learnSitePConv",args)
        elif args.learnSitePKConv:
            model = modelBuild("learnSitePKConv",args)
        elif args.branchLearnSitePConv:
            model = modelBuild("branch_lSitePConv",args)
        elif args.sharePConv_lSite:
            model = modelBuild("sharePConv_lSite",args)
        else:# 既存法(PConvUNet)
            # モデルのビルド
            model = modelBuild("pconv",args)
        
        # データセット
        trainData = (trainMasked, trainMask, trainImg)
        validData = ((validMasked, validMask), validImg)
        test_sample = (testMasked[0:1], testMask[0:1])

    # モデルのコンパイル(マスクは１種類のみの設定しかできない)
    ## TODO:今後改良する必要がある
    mask = trainMask[0:1]
    mask = mask.astype('float32') if mask.dtype != np.float32 else mask
    model.compile(args.lr, mask)

    checkpoint_path = f"{experiment_path}{os.sep}logs{os.sep}cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    # 学習中にテストを行いプロットするクラス
    class testSample(tf.keras.callbacks.Callback):
        def __init__(self, test_sample, gt_sample, testpath="",):
            super(testSample, self).__init__()
            self.sample = test_sample
            self.siteFs = []
            self.testpath = testpath
            plt.imshow(gt_sample[0,:,:,0])
            plt.colorbar()
            plt.savefig(f"{testpath}{os.sep}test_gt_sample.png")

        def on_epoch_end(self, epoch, logs=None):
            pred_img = model.predict_step(self.sample)
            plt.clf()
            plt.close()
            plt.imshow(pred_img[0,:,:,0])
            plt.colorbar()
            plt.savefig(f"{self.testpath}{os.sep}pred_{epoch}.png")

            if args.learnSitePConv:
                model.plotSiteFeature(epoch,f"{self.testpath}")
            elif args.branchLearnSitePConv:
                model.plotSiteFeature(epoch,f"{self.testpath}",plotfmap=True)
                self.siteFs.append(model.getSiteFeature())
        
        def on_train_end(self,logs=None):
            pickle.dump(self.siteFs,open(f"{testpath}{os.sep}training_siteFeatures.png","wb"))
            return


    # コールバック関数の設定
    ## EarlyStoppingを導入
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_PSNR', min_delta=0.1, patience=args.patience, 
            verbose=1, mode='max', baseline=None, restore_best_weights=True
        ),
        testSample(test_sample, testImg[0:1], testpath=test_path)
    ]

    
    model.changeTrainPhase(args.phase) # フェーズ切り替え
    if args.phase > 1:
        # フェーズ0や1の情報を参照
        checkpoint_path = f"{experiment_path}{os.sep}logs{os.sep}cp.ckpt"
        model.load_weights(checkpoint_path)

    # =======================
    # 学習
    history = model.fit(
        trainData,
        batch_size=batchsize,
        validation_data=validData,
        epochs=args.epochs,
        validation_split=0.1,
        callbacks=callbacks)

    # =======================
    # 損失のプロット
    history = history.history
    pickle.dump(history, open(f"{loss_path}{os.sep}trainingHistory.pickle","wb"))

    metrics = ['loss','PSNR','loss_hole','loss_valid','loss_tv']
    # if args.learnSitePConv or args.branchLearnSitePConv:
    #     metrics.append('loss_L1site')
    traLoss = [history[met] for met in metrics]
    valLoss = [history['val_'+met] for met in metrics]

    for i,met in enumerate(metrics):
        print(f"plot {met}")
        tL = traLoss[i]
        vL = valLoss[i]
        plotDatas([tL,vL],title=met,
            savepath=f"{loss_path}{os.sep}training_validation_{met}.png",
            ylabel=met,labels=[met, 'val_'+met],
            style=['bo-','r'])

    # =======================

