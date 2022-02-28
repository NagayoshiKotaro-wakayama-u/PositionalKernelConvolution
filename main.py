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
    TRAIN_PICKLE = dspath+"train.pickle" # 辞書Object(key:"images","labels")
    TRAIN_MASK_PICKLE = dspath+"train_mask.pickle" # np.array()
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
    # steps_per_epoch = train_Num//batchsize # 1エポック内のiteration数

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

    # pdb.set_trace()
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
        elif args.branchLearnSitePKConv: # branch and PKConv
            model = modelBuild("branch_lSitePKConv",args)
        elif args.learnSitePKConv:
            model = modelBuild("learnSitePKConv",args)
        elif args.branchLearnSitePConv:# 最新手法
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

    checkpoint_path = f"{log_path}{os.sep}cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)


    # 学習中にテストを行いプロットするクラス
    class testSample(tf.keras.callbacks.Callback):
        def __init__(self, test_sample, gt_sample, testpath="",):
            super(testSample, self).__init__()
            self.sample = test_sample
            # self.siteFs = []
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
        
    class phaseSwitchCycle(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            # エポック数
            self.totalEpoch = 0
            self.countEpoch = 0

            # phase は　1 と 2を繰り返す
            self.pind = 0
            if args.branchLearnSitePKConv or args.branchLearnSitePConv:
                self.phases = [1,4,5]
            else:
                self.phases = [1,2]
            self.phaseNum = len(self.phases)
            self.cycleNums = [1 for _ in range(self.phaseNum)] # 学習のサイクルは１Epochずつ
        
        def on_epoch_end(self, epoch, logs=None):
            if self.totalEpoch == 0:
                # 初めは両方更新しないとパラメータが更新されない？
                model.changeTrainPhase(0)
            else:
                self.countEpoch += 1
                cycleNum = self.cycleNums[self.pind]
                
                if self.countEpoch == cycleNum:
                    self.countEpoch = 0
                    # 次のフェーズへ
                    self.pind = (self.pind+1)%self.phaseNum
                    phase = self.phases[self.pind]
                    model.changeTrainPhase(phase)

            self.totalEpoch += 1
            
    class earlySiteFeatureStop(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            # エポック数
            self.phase2Counter = 0
            self.phase = 1
            self.maxSiteLearningEpoch = 5

        def on_epoch_end(self, epoch, logs=None):
            # 位置特性の学習をした回数をカウント
            if self.phase == 2:
                self.phase2Counter += 1

            if self.phase2Counter > self.maxSiteLearningEpoch:
                # 一定数位置特性の学習をしたらもうU-Netだけしか学習しない
                self.phase = 1
                model.changeTrainPhase(1)
                # pdb.set_trace()
            else:
                # フェーズ切り替え(1と2)
                self.phase = (self.phase % 2) + 1 # 0か1にして＋１
                model.changeTrainPhase(self.phase)
            


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

    # 
    if args.switchingTrainPhase:
        callbacks.append(phaseSwitchCycle())
        model.changeTrainPhase(0)
    elif args.earlySiteFeatureStop: # 最新手法
        callbacks.append(earlySiteFeatureStop())
        model.changeTrainPhase(0) # 初めは両方のモデルを同時に学習（phase = 0に設定）

    
    # if args.branchLearnSitePConv:
    #     model.changeTrainPhase(args.phase) # フェーズ切り替え
    #     # 1つ前のフェーズ
    #     current_ph = ""
    #     if args.phase - 1 > 1: # 2以降は番号を付ける
    #         current_ph = f"{args.phase - 1}"

    #     if args.pretrainModel != "":
    #         # 1つ前のフェーズのパラメータをロード
    #         expath = f".{os.sep}experiment{os.sep}{args.pretrainModel}_logs"
    #         load_checkpoint_path = f"{expath}{os.sep}logs{current_ph}{os.sep}cp.ckpt"
    #         model.load_weights(load_checkpoint_path)

    # =======================
    # 学習
    # pdb.set_trace()
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

