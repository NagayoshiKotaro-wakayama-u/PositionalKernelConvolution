import tensorflow as tf
import matplotlib.pylab as plt
import os
import sys
import pickle
import numpy as np
import pdb
import copy
from argparse import ArgumentParser
from libs.util import loadSiteImage,PSNR,plotImgs,nonhole,cmap,clip,extractFromTop
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from libs.modelConfig import parse_args,modelBuild
from libs.models import PConvLearnSite

if __name__ == "__main__":
    args = parse_args(isTrain=False)
    tf.config.run_functions_eagerly(True)

    # フェーズ
    ph = ""
    if args.phase > 1:
        ph = f"{args.phase}"

    site_path = f"data{os.sep}siteImage{os.sep}"
    experiment_path = ".{0}experiment{0}{1}_logs".format(os.sep,args.experiment)
    img_w = args.imgw
    img_h = args.imgh
    dspath = ".{0}data{0}{1}{0}".format(os.sep,args.dataset)
    
        
    # 出力先のディレクトリを作成
    result_path = f"{experiment_path}{os.sep}result"
    test_path = f"{result_path}{os.sep}test"
    comp_path = f"{result_path}{os.sep}comparison"
    siteConv_path = f"{result_path}{os.sep}site"

    for DIR in [result_path, test_path, comp_path, siteConv_path]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)
    
    # 各pickleデータのパス
    TEST_PICKLE = dspath+"test.pickle"
    TEST_MASK_PICKLE = dspath+"test_mask.pickle"

    # テストデータの読み込み
    testImg = pickle.load(open(TEST_PICKLE,"rb"))["images"]
    testLabel = pickle.load(open(TEST_PICKLE,"rb"))["labels"]
    testMask = pickle.load(open(TEST_MASK_PICKLE,"rb"))
    testMasked = testImg*testMask

    # ================
    # トイデータの並び替え
    if "stripe-rect" in args.dataset:
        # testImg = [img for img in testImg]
        testLabel = [int(lab) for lab in testLabel]
        testLabel = np.array(testLabel)
        idx = np.argsort(testLabel)
        testImg = testImg[idx]
        testLabel = testLabel[idx]
        testMasked = testMasked[idx]
    # ================

    # データ名にquakeとある場合はsea.pngを利用
    existPath = f"data{os.sep}sea.png" if "quake" in args.dataset else ""
    exist = np.array(Image.open(existPath))/255 if existPath!="" else np.ones(testImg.shape[1:3])
    
    # データ数
    dataNum = testImg.shape[0]

    # モデルのビルド
    if args.PKConv:
        model = modelBuild("pkconvAndSiteCNN",args,isTrain=False)

        # 位置特性の読み込み(loadSiteImage内で正規化)
        siteImg = loadSiteImage(
            [site_path + s for s in args.sitePath],
            smax=args.siteMax,
            smin=args.siteMin
        )
        siteImgs = np.tile(siteImg,(testImg.shape[0],1,1,1))
        dataset = (testMasked, testMask, siteImgs, testImg)

    else:
        if args.learnSitePConv:
            model = modelBuild("learnSitePConv",args)
        elif args.learnSitePKConv:
            model = modelBuild("learnSitePKConv",args)
        elif args.branchLearnSitePConv:
            model = modelBuild("branch_lSitePConv",args)
        else:
            model = modelBuild("pconv",args,isTrain=False)
        dataset = (testMasked, testMask, testImg)
    
    model.compile(args.lr, testMask[0:1])
    checkpoint_path = f"{experiment_path}{os.sep}logs{ph}{os.sep}cp.ckpt"
    model.load_weights(checkpoint_path)

    # 学習した位置特性で値を持つ点付近の誤差
    #==============================================
    mask = testMask[0,:,:,0]

    # pdb.set_trace()
    if isinstance(model,PConvLearnSite):
        model.plotSiteFeature(epoch="-test",plotSitePath=siteConv_path)
        # pdb.set_trace()
        # sys.exit()
    if False:
        # TODO:後で消す
        # テスト用(上位10％だけプロットする)
        originalSiteFeature = model.getSiteFeature()
        # originalSiteFeature = model.getSiteFeature()[0]
        # pdb.set_trace()
        # initValue = model.initValues[0]
        initValue = model.initValue
        if args.flatSiteFeature:
            model.setSiteFeature(initValue)

        origin_s = originalSiteFeature[0,:,:,0]
        absSite = np.abs(origin_s-initValue[0,:,:,0])

        nonZeroSite = absSite[absSite>0]
        nonZeroSitePos = np.where(absSite>0)

        ratioFromTop = 0.1
        highSiteInds = extractFromTop(nonZeroSite,ratioFromTop) # 上位何割かの座標を取得(昇順)

        lowLimitInd = highSiteInds[0] # 上位何割か点のIndex
        threshValue = nonZeroSite[lowLimitInd] # 上位何割か点の値（境界の値）

        _,site01 = cv2.threshold(absSite,threshValue,1,cv2.THRESH_BINARY) # 境界値より大きい位置特性をマスク化
        # プロット用位置特性（高い値を持つ点）
        plotHighSite = initValue + (originalSiteFeature - initValue)*site01[np.newaxis,:,:,np.newaxis]
        model.setSiteFeature(plotHighSite) # 位置特性の値を上位何割かだけ残した状態にする
        _ = model.plotSiteFeature(plotSitePath=siteConv_path, saveName=f"top{ratioFromTop}SiteMask")
        model.setSiteFeature(originalSiteFeature) # 位置特性の値を元に戻す
    # else:
    site01 = mask
    # pdb.set_trace()
    ksize = 7
    dilated_site01 = cv2.dilate(site01,np.ones((ksize,ksize),np.uint8))# maskを拡大
    _,dilated_site01 = cv2.threshold(dilated_site01,1e-10,1,cv2.THRESH_BINARY)# 二値化
    dilated_site01 = dilated_site01 * exist - site01 # 海洋部・観測点は除く  
    #==============================================


    preds, errs, MAEs, MSEs, PSNRs, obsAroundPSNRs = [], [], [], [], [], []
    ite = 0
    cm_bwr = plt.get_cmap("bwr") # カラーマップ
    # テストの実行と評価
    for data in zip(*dataset):
        print(f"\r progress : {ite}/{dataNum}",end="")

        lab = testLabel[ite]
        # トイデータである場合はラベルを変換
        if "gaus" in args.dataset:
            lab = "upper" if lab=="2" else "below"
        
        # データの取得
        data = tuple([d[np.newaxis,:,:,:] for d in data])
        masked = data[0]
        mask = data[1]
        if args.PKConv:
            site = data[2]
            img = data[3]
            data = (masked, mask, site)
            if ite==0: # 初回のみ位置特性を出力
                _ = model.build_pkconv_unet(masked, mask, site, training=False, plotSitePath=siteConv_path)
        else:
            if args.learnSitePConv:
                if ite==0: # 初回のみ出力
                    _ = model.plotSiteFeature(plotSitePath=siteConv_path)

            img = data[2]
            data = (masked,mask)
    
        # 予測
        pred = model.predict_step(data)
        # numpy arrayじゃないならnumpyにする
        if not isinstance(pred, np.ndarray):
            pred = pred.numpy()
        pred = np.squeeze(pred)
        preds.append(pred)

        img = np.squeeze(img)
        mask = np.squeeze(mask)
        masked = np.squeeze(masked)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 誤差の計算
        err = pred - img
        errs.append(err)
        # MAE
        mae = np.mean(np.abs(err))
        MAEs.append(mae)
        # MSE
        mse = np.mean(np.square(err))
        MSEs.append(mse)
        # PSNR
        psnr = PSNR(pred,img,exist=exist)
        PSNRs.append(psnr)

        # pdb.set_trace()
        # 値を持つ点付近以外を消去
        obsAroundTruth = img * dilated_site01
        obsAroundPred = pred * dilated_site01
        # 値を持つ点付近のPSNR
        obsAroundPSNR = PSNR(obsAroundPred, obsAroundTruth, exist=dilated_site01)
        obsAroundPSNRs.append(obsAroundPSNR)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~

        #=======================================
        # 画像のプロット
        ## テスト結果
        if args.plotTest:
            tmp = (pred*255).astype("uint8")
            cv2.imwrite(f"{test_path}{os.sep}test{ite}_{lab}.png", tmp[0,:,:,0])
        
        ## 詳細な比較
        if args.plotComparison:
            # 3ｘ3のプロット
            _, axes = plt.subplots(3, 3, figsize=(16, 15))
            # 各データのタイトル
            titles = [
                "masked",
                f"pred(MAE={mae:.3f},PSNR={psnr:.2f})",
                "origin"
                ]

            ## カラー画像用にtile
            mask_rgb = np.tile(mask[:,:,np.newaxis],(1,1,3))
            exist_rgb = np.tile(exist[:,:,np.newaxis],(1,1,3))

            ### 入力・予測・真値の比較
            # pdb.set_trace()
            x1 = cmap(masked)
            x1[mask_rgb==0] = 255
            xs = [x1,cmap(pred),cmap(img)]
            for i,x in enumerate(xs):
                x[exist_rgb==0] = 255
                axes[0,i].imshow(x,vmin=0,vmax=255)
                axes[0,i].set_title(titles[i])

            ### ヒストグラム
            bins = 20
            hs = []
            nonh_img = nonhole(img,exist)
            nonh_pred = nonhole(pred,exist)
            hs.append(nonhole(img,mask*exist))
            hs.append(nonh_pred)
            hs.append(nonh_img)
            tmp = np.concatenate(hs,axis=0)
            maxs = np.max(tmp)

            for i,h in enumerate(hs):
                axes[1,i].hist(h,bins=bins,range=(0,maxs))

            ### 真値(横)・予測値(縦)のプロット
            axes[2,0].scatter(nonh_img, nonh_pred)
            axes[2,0].set_xlabel("true value")
            axes[2,0].set_ylabel("pred value")

            ### 誤差マップ
            err = cm_bwr(clip(err,-0.3,0.3))[:,:,:3] # -0.1~0.1の偏差をカラーに変換
            axes[2,1].imshow(err*exist_rgb,vmin=0,vmax=1.0)
            
            plt.savefig(f"{comp_path}{os.sep}comparison{ite}.png")
            plt.close()
            
        #=======================================
        ite += 1


    # pdb.set_trace()
    # 保存するデータ
    summary_data = {
        "preds":preds,
        "error":errs,
        "MAE":np.mean(MAEs),
        "MSE":np.mean(MSEs),
        "PSNR":np.mean(PSNRs),
        "MAEs":MAEs,
        "MSEs":MSEs,
        "PSNRs":PSNRs,
        "obsAroundPSNRs":obsAroundPSNRs,
    }
    # pdb.set_trace()
    print(f"obsAround:{np.mean(obsAroundPSNRs)}")

    # 保存
    if args.flatSiteFeature:
        pickle.dump(summary_data, open(f"{result_path}{os.sep}testSummary_flatSiteFeature.pickle","wb"))
    else:
        pickle.dump(summary_data, open(f"{result_path}{os.sep}testSummary.pickle","wb"))

  
