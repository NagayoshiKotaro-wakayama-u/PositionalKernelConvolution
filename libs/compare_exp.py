import pickle
import numpy as np
from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
from PIL import Image
import pdb
import cv2
from util import cmap,clip,nonhole,PSNR,rangeError,resultInpaint,SqueezedNorm

def parse_args():
    parser = ArgumentParser(description='Test script for PConv inpainting')
    parser.add_argument('dataset',type=str, help="データセット名")
    parser.add_argument('experiments',type=lambda x:list(map(str,x.split(","))),default='', )
    parser.add_argument('-titles','--titles',type=lambda x:list(map(str,x.split(","))))
    parser.add_argument('-comp','--comparison',action='store_true')
    parser.add_argument('-cv','--crossVaridation',type=int,default=None)
    parser.add_argument('-spath','--savePath',type=str,default=None)
    parser.add_argument('-region','--regionPlot',action='store_true')
    parser.add_argument('-flatSite','--flatSiteFeature',action='store_true')
    parser.add_argument('-phase','--phase',type=int,default=0)
    parser.add_argument('-siteComp','--siteComp',action='store_true')
    return  parser.parse_args()

# 領域の左上座標(y1,x1)と右下座標(y2,x2)が[y1, x1, y2, x2]の順に並ぶ
regions = {
    "osaka":[240,230,285,280],
    "tokyo":[165,390,215,440],
    "nagoya":[200,290,280,340],
}

# (y1,x1) (y2,x2)
# regions = {
#     "osaka":[224, 250, 279, 267],
#     "tokyo":[295, 395, 310, 435],
#     "nagoya":[235, 306, 292, 352]
# }


# 比較プロットの作成を行うプログラム
if __name__ == "__main__":
    args = parse_args()

    if args.savePath != None:
        titlesJoin = args.savePath
    elif args.titles != None:
        titlesJoin = "".join(args.titles)
    else:
        titlesJoin = "".join(args.experiments)

    # 保存先ディレクトリ
    savepath = f".{os.sep}fig{os.sep}{titlesJoin}"
    for DIR in [savepath]:
        if not os.path.isdir(DIR):
            os.makedirs(DIR)
    # PSNRなどの情報を出力するテキストファイル
    txtfile = open(f"{savepath}{os.sep}comparison.txt","w")

    if args.crossVaridation is not None:
        # タイトル        
        if args.titles != None:
            titlesJoin = "".join(args.titles)
        else:
            titlesJoin = "".join(args.experiments)

        # データ名にquakeとある場合はsea.pngを利用
        existPath = f"data{os.sep}sea.png" if "quake" in args.dataset else ""
        # 比較数
        compNum = len(args.experiments)
        
        PSNRsList = []
        obsAroundPSNRList = [[[] for _ in range(2)] for _ in range(compNum)]
        if args.siteComp: # 位置特性の比較を行うかどうか
            site_aves = [0 for _ in range(compNum)]

        # 交差検証
        for eind in range(compNum):
            avePSNR = 0  
            exp = args.experiments[eind]       
            PSNRs,osakaPSNR,tokyoPSNR,nagoyaPSNR = [],[],[],[]
            
            ph = "" # フェーズ
            if args.phase > 1: # フェーズが1以上でディレクトリ名が変わる
                ph = args.phase

            all_cv_preds = []
            all_cv_imgs = []
            all_cv_PSNRs = []
            
            for i in range(args.crossVaridation):
                experiment_path = f"experiment{os.sep}{exp}_cv{i}_logs"
                result_path = f"{experiment_path}{os.sep}result{ph}"
                summary_path = f"{result_path}{os.sep}testSummary.pickle"

                summaryData = pickle.load(open(summary_path,"rb"))
                all_cv_preds.append(summaryData["preds"])
                all_cv_PSNRs.append(summaryData["PSNRs"])

                # データのロード
                exist = np.array(Image.open(existPath))/255 if existPath!="" else np.ones(testImg.shape[1:3])

                # 位置特性の比較のみを行う場合
                if args.siteComp:
                    site_path = f"{result_path}{os.sep}site"
                    siteFeature = pickle.load(open(f"{site_path}{os.sep}siteFeature.pickle","rb"))
                    site_aves[eind] += siteFeature[0] if isinstance(siteFeature,list) else siteFeature
                else:
                    # 各pickleデータのパス
                    dspath = f"data{os.sep}{args.dataset}-crossVaridation{i}{os.sep}"
                    TEST_PICKLE = dspath+"test.pickle"
                    TEST_MASK_PICKLE = dspath+"test_mask.pickle"
                    # テストデータの読み込み
                    testImg = pickle.load(open(TEST_PICKLE,"rb"))["images"]
                    testLabel = pickle.load(open(TEST_PICKLE,"rb"))["labels"]
                    testMask = pickle.load(open(TEST_MASK_PICKLE,"rb"))
                    testMasked = np.squeeze(testImg*testMask)
                    mask = testMask[0,:,:,0]
                    all_cv_imgs.append(np.squeeze(testImg))

                    # テスト結果の読み込み
                    # psnr = summaryData["PSNR"]
                    # avePSNR += psnr

                    resultPlot = resultInpaint(testImg, mask, exist, experiment_path, phase=args.phase, regions=regions)
                    pdb.set_trace()
                    
                    # 各cross validation全体のPSNRを計算
                    psnr = resultPlot.allPSNR()
                    
                    PSNRs.append(psnr) # each fold, all pixel
                    avePSNR += psnr
                    regdic = resultPlot.allRegionPSNR()
                    osakaPSNR.append(regdic["osaka"])
                    tokyoPSNR.append(regdic["tokyo"])
                    nagoyaPSNR.append(regdic["nagoya"])

                    obsAroundPSNR = np.mean(summaryData["obsAroundPSNRs"])
                    obsAroundPSNRList[eind][0].append(obsAroundPSNR)

                    # 位置特性の影響度（位置特性周辺のPSNR）を比較
                    if args.flatSiteFeature:
                        summary_flatSite_path = f"{result_path}{os.sep}testSummary_flatSiteFeature.pickle"
                        summaryData_flatSite = pickle.load(open(summary_flatSite_path,"rb"))
                        obsAroundPSNR_flatSite = np.mean(summaryData_flatSite["obsAroundPSNRs"])

                        obsAroundPSNRList[eind][1].append(obsAroundPSNR_flatSite)
                    
            #pdb.set_trace()
            all_cv_preds = np.concatenate(all_cv_preds) # shape [150, 512, 512]
            all_cv_imgs = np.concatenate(all_cv_imgs) # shape [150, 512, 512]
            
            # calc ALL cross validation PSNR
            all_resultPlot = resultInpaint(all_cv_imgs, mask, exist, experiment_path, phase=args.phase, regions=regions, preds=all_cv_preds)
            all_pixel_regdic = resultPlot.allRegionPSNR()
            all_pixel_osakaPSNR = all_pixel_regdic["osaka"]
            all_pixel_tokyoPSNR = all_pixel_regdic["tokyo"]
            all_pixel_nagoyaPSNR = all_pixel_regdic["nagoya"]
            all_cv_pixel_PSNR = PSNR(all_cv_preds, all_cv_imgs, exist=np.tile(exist[np.newaxis],[all_cv_imgs.shape[0],1,1]))
            
            # over image average PSNR
            over_img_avePSNR = np.mean(np.concatenate(all_cv_PSNRs))
            regOverimage_resultPlot = resultInpaint(all_cv_imgs, mask, exist, experiment_path, phase=args.phase, regions=regions, preds=all_cv_preds, isRegionOver_image=True)
            overimage_regdic = regOverimage_resultPlot.allRegionPSNR()
            overimage_osakaPSNR = overimage_regdic["osaka"]
            overimage_tokyoPSNR = overimage_regdic["tokyo"]
            overimage_nagoyaPSNR = overimage_regdic["nagoya"]

            
            if args.siteComp:# 交差検証各回の位置特性を平均する
                site_aves[eind] = site_aves[eind]/args.crossVaridation
            else:
                avePSNR = avePSNR/args.crossVaridation
                aveOsakaPSNR = np.mean(osakaPSNR)
                aveTokyoPSNR = np.mean(tokyoPSNR)
                aveNagoyaPSNR = np.mean(nagoyaPSNR)
                # pdb.set_trace()
                AveObsAroundPSNR = np.mean(obsAroundPSNRList[eind][0])
                # print(f"{exp}:avePSNR={avePSNR}, PSNRs={PSNRs}")
                #txtfile.write(f"{exp}:\navePSNR={avePSNR}\n osaka :{aveOsakaPSNR}\n tokyo :{aveTokyoPSNR}\n nagoya:{aveNagoyaPSNR}\nobAround:{AveObsAroundPSNR}\nPSNRs={PSNRs}\n\n")
                txtfile.write(f"{exp}:\navePSNR(over fold)={avePSNR}\n osaka(over fold) :{aveOsakaPSNR}\n tokyo(over fold) :{aveTokyoPSNR}\n nagoya(over fold):{aveNagoyaPSNR}\nobAround:{AveObsAroundPSNR}\nPSNRs={PSNRs}\n")
                txtfile.write(f"PSNR(all pixel)={all_cv_pixel_PSNR}\n osaka(all pixel) :{all_pixel_osakaPSNR}\n tokyo(all pixel) :{all_pixel_tokyoPSNR}\n nagoya(all pixel):{all_pixel_nagoyaPSNR}\n")
                txtfile.write(f"avePSNR(over image)={over_img_avePSNR}\n osaka(over image) :{overimage_osakaPSNR}\n tokyo(over image) :{overimage_tokyoPSNR}\n nagoya(over image):{overimage_nagoyaPSNR}\n\n")

                PSNRsList.append(PSNRs)

        if args.siteComp: # 位置特性の比較（差分を描画）
            site01_dist = site_aves[0] -site_aves[1]
            cmbwr = plt.get_cmap('bwr')
            cmbwr.set_under('black')

            _min = min([np.min(_s) for _s in site_aves])
            _max = max([np.max(_s) for _s in site_aves])

            # 各回の平均位置特性をプロット
            for site,exp in zip(site_aves,args.experiments):
                plt.clf()
                plt.close()
                plt.imshow(site[0,:,:,0],cmap=cmbwr,norm=SqueezedNorm(vmin=_min,vmax=_max,mid=0),interpolation='none')
                plt.colorbar(extend='both')
                plt.savefig(f"{savepath}{os.sep}site-ave_{exp}.png")   

            # 位置特性の差をプロット
            smin = np.min(site01_dist)
            smax = np.max(site01_dist)
            vrange = abs(smax-smin)
            # 上位20％の誤差のみプロット
            # mask20per = np.ones(site01_dist.shape)
            # mask20per[site01_dist < smax - vrange*0] = 0
            # mask20per[site01_dist > smin] = 1
            # site01_dist = site01_dist * mask20per
            # site01_dist[0,:,:,0] = -10000
            plt.clf()
            plt.close()
            plt.imshow(
                site01_dist[0,:,:,0],
                cmap=cmbwr,
                norm=SqueezedNorm(vmin=smin,vmax=smax,mid=0),
                interpolation='none'
                )
            plt.title("siteComparison_0-1")
            plt.colorbar(extend='both')
            plt.savefig(f"{savepath}{os.sep}siteComparison.png")
        
        else:    
            plt.close()
            for eind,psnrs in enumerate(PSNRsList):
                plt.plot(range(args.crossVaridation),psnrs,label=args.experiments[eind])
            plt.legend()
            plt.xlabel("crossVaridate iteration")
            plt.ylabel("PSNR")
            plt.savefig(f"{savepath}{os.sep}PSNRs_crossvaridation.png")

        if args.flatSiteFeature:
            for eind in range(compNum):
                plt.clf()
                plt.close()
                plt.scatter(range(10),obsAroundPSNRList[eind][0],label="with site") # 位置特性ありの周辺誤差
                plt.scatter(range(10),obsAroundPSNRList[eind][1],label="without site") # 位置特性なしの周辺誤差
                plt.scatter(range(10),PSNRsList[eind],label="PSNR(field)")
                plt.legend()
                plt.xlabel("crossVaridate iteration")
                plt.ylabel("PSNR")
                plt.savefig(f"{savepath}{os.sep}siteAroundPSNRs_comp{eind}.png")




    else:

        # 地域のプロット
        if args.regionPlot:
            reg_paths = [f"{savepath}{os.sep}{key}" for key in regions]
            for DIR in reg_paths:
                if not os.path.isdir(DIR):
                    os.makedirs(DIR)

        dspath = ".{0}data{0}{1}{0}".format(os.sep,args.dataset)
        # 各pickleデータのパス
        TEST_PICKLE = dspath+"test.pickle"
        TEST_MASK_PICKLE = dspath+"test_mask.pickle"

        # テストデータの読み込み
        testImg = pickle.load(open(TEST_PICKLE,"rb"))["images"]
        testLabel = pickle.load(open(TEST_PICKLE,"rb"))["labels"]
        testLabel = [lab.split(".")[0] for lab in testLabel]
        # pdb.set_trace()
        testMask = pickle.load(open(TEST_MASK_PICKLE,"rb"))
        testMasked = np.squeeze(testImg*testMask)
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

        experiments = args.experiments
        # フェーズがある場合は一つの実験について各フェーズを比較
        # if args.phase > 0:
        #     experiments = experiments[:1]*args.phase
            # args.titles = [f"phase{ph+1}" for ph in range(args.phase)]
        
        experiment_paths = [f".{os.sep}experiment{os.sep}{ex}_logs" for ex in experiments]
        loss_paths = [f"{exp}{os.sep}losses" for exp in experiment_paths]
        
        # 予測値を取得
        preds,psnrs,epochs,siteFeatures = [],[],[],[]
        for i,exp in enumerate(experiment_paths):
            
            ph = "" # フェーズ
            if args.phase > 1: # フェーズ2以降
                ph = args.phase
            
            result_path = f"{exp}{os.sep}result{ph}"
            summary_path = f"{result_path}{os.sep}testSummary.pickle"
            loss_path = f"{exp}{os.sep}losses{ph}"

            # 位置特性の比較を行う場合
            if args.siteComp:
                site_path = f"{result_path}{os.sep}site"
                siteFeature = pickle.load(open(f"{site_path}{os.sep}siteFeature.pickle","rb"))
                siteFeatures.append(siteFeature[0] if isinstance(siteFeature,list) else siteFeature)
            else:
                summaryData = pickle.load(open(summary_path,"rb"))
                # 予測値等を取得後、shapeを絞る
                # pdb.set_trace()
                pred = np.squeeze(np.array(summaryData["preds"])) # shape=[n,h,w]
                preds.append(pred)
                psnr = PSNR(pred, testImg[:,:,:,0], exist=np.tile(exist[np.newaxis,:,:], [testImg.shape[0],1,1]))
                # psnr = summaryData["PSNR"]
                psnrs.append(psnr)
                # loss = pickle.load(open(f"{loss_path}{os.sep}trainingHistory.pickle","rb"))["loss"]
                # epochs.append(len(loss))
        
        if args.siteComp:# 位置特性の比較（差分を描画）
            site01_dist = siteFeatures[0] -siteFeatures[1]
            cmbwr = plt.get_cmap('bwr')
            cmbwr.set_under('black')

            _min = min([np.min(_s) for _s in siteFeatures])
            _max = max([np.max(_s) for _s in siteFeatures])

            # 各実験の位置特性をプロット
            for site,exp in zip(siteFeatures,args.experiments):
                plt.clf()
                plt.close()
                plt.imshow(site[0,:,:,0],cmap=cmbwr,norm=SqueezedNorm(vmin=_min,vmax=_max,mid=0),interpolation='none')
                plt.colorbar(extend='both')
                plt.savefig(f"{savepath}{os.sep}site-ave_{exp}.png")   

            # 位置特性の差をプロット
            smin = np.min(site01_dist)
            smax = np.max(site01_dist)
            vrange = abs(smax-smin)
            plt.clf()
            plt.close()
            plt.imshow(
                site01_dist[0,:,:,0],
                cmap=cmbwr,
                norm=SqueezedNorm(vmin=smin,vmax=smax,mid=0),
                interpolation='none'
                )
            plt.title("siteComparison")
            plt.colorbar(extend='both')
            plt.savefig(f"{savepath}{os.sep}siteComparison.png")
            pdb.set_trace()

        compNum = len(experiments)
        
        # 各データのプロット
        # pdb.set_trace()
        images = [testMasked]+preds+[testImg]
        if args.titles!=None:
            titles = ["input"]+args.titles+["original"]
        else:
            titles = ["input"]+experiments+["original"]

        # テスト結果等の出力
        for i,title in enumerate(titles[1:-1]):
            print(f"{title}:PSNR={psnrs[i]}")
            txtfile.write(f"{title}:PSNR={psnrs[i]}")
            # print(f"{title}:PSNR={psnrs[i]}, epoch={epochs[i]}")
            # txtfile.write(f"{title}:PSNR={psnrs[i]}, epoch={epochs[i]}")

        mask = testMask[0] # shape=[h,w,1]
        cm_bwr = plt.get_cmap("bwr") # 青⇒白⇒赤のカラーマップ

        # obsAround_preds = [[] for _ in range(compNum+2)]
        # obsAround_imgs = []
        # predicts = [[] for _ in range(compNum+2)]
        # truths = []
        # 観測点周辺の誤差計算用（位置特性の影響度調査のため）
        # ksize = 3
        # dilated_mask = cv2.dilate(mask[:,:,0],np.ones([ksize,ksize]))
        # dilated_mask = dilated_mask*exist - mask[:,:,0]
        # obsAround_img = nonhole(_truth,dilated_mask)
        # obsAroundRange = [np.min(obsAround_img),np.max(obsAround_img)]
        # obsAround_imgs.append(obsAround_img)

        def comparisonPlot(xs,ex=exist,msk=mask,dirpath=savepath,savename="comp"):
            for ind in range(xs[-1].shape[0]):
                print(f"\r progress : {ind}/{xs[-1].shape[0]}",end="")

                # 最後尾が真値
                # pdb.set_trace()
                _truth = np.squeeze(xs[-1][ind])
                nonh_img = nonhole(_truth,ex)
                # truths.append(nonh_img)
                xmax = max([np.max(x[ind]) for x in xs[1:]])
                xmax = xmax*1.05
                xmin = min([np.min(x[ind]) for x in xs[1:]])

                ## カラー画像用にtile
                mask_rgb = np.tile(msk,(1,1,3)) # shape=[h,w,3]
                exist_rgb = np.tile(ex[:,:,np.newaxis],(1,1,3)) # shape=[h,w,3]

                plt.clf()
                plt.close()
                _, axes = plt.subplots(3, compNum+2, figsize=(5*(compNum+2), 15))
                
                for i in range(compNum+2):
                    #==================================
                    # 1行目（入力・予測値・真値）
                    ## カラー画像に変換
                    x = np.squeeze(cmap(xs[i][ind]))
                    if i==0:
                        x[mask_rgb==0] = 255 # マスク部分を白に
                    if "quake" in args.dataset:
                        x[exist_rgb==0] = 255 # 

                    axes[0,i].imshow(x)
                    axes[0,i].set_title(titles[i])
                    #==================================
                    # 2行目
                    ## 真値・予測値プロット
                    _pred = np.squeeze(xs[i][ind])
                    nonh_pred = nonhole(_pred,ex)
                    axes[1,i].scatter(nonh_img, nonh_pred)
                    axes[1,i].set_xlim(xmin,xmax)
                    axes[1,i].set_ylim(xmin,xmax)
                    if i==0:
                        axes[1,i].set_xlabel("true value")
                        axes[1,i].set_ylabel("pred value")
                    #==================================
                    # 3行目
                    ## 誤差マップ                    
                    err = _pred - _truth
                    if i != 0 and i != compNum + 1:
                        err = cm_bwr(clip(err,-0.3,0.3))[:,:,:3]
                        axes[2,i].imshow(err)
                        axes[2,i].set_title(f"PSNR:{PSNR(_pred,_truth,exist=ex)}")
                    #==================================
                    # 4行目
                    ## 観測点周辺の真値・予測値プロット
                    # obsAround_pred = nonhole(_pred,dilated_mask)
                    # axes[3,i].scatter(obsAround_img, obsAround_pred)
                    # axes[3,i].set_ylim(obsAroundRange[0],obsAroundRange[1])
                    # if i==0:
                    #     axes[3,i].set_xlabel("true value")
                    #     axes[3,i].set_ylabel("pred value")
                    #==================================
                    # obsAround_preds[i].append(obsAround_pred)
                    # predicts[i].append(nonh_pred)

                plt.savefig(f"{dirpath}{os.sep}{savename}{ind}_{testLabel[ind]}.png")

        if args.comparison:
            comparisonPlot(images)
        
        if args.regionPlot:
            for rind,key in enumerate(list(regions)):
                r = regions[key]
                regImgs = [np.squeeze(img)[:,r[0]:r[2],r[1]:r[3]] for img in images]
                regExist = exist[r[0]:r[2],r[1]:r[3]]
                regMask = mask[r[0]:r[2],r[1]:r[3]]
                comparisonPlot(
                    regImgs,
                    ex=regExist,
                    msk=regMask,
                    dirpath=reg_paths[rind],
                    savename=f"region-{key}"
                )


        # pdb.set_trace()
        # obsAround_preds = [np.concatenate(preds) for preds in obsAround_preds]
        # obsAround_imgs = np.concatenate(obsAround_imgs)
        
        # predicts = [np.concatenate(preds) for preds in predicts]
        # truths = np.concatenate(truths)

        # plt.clf()
        # plt.close()
        # _, axes = plt.subplots(1, compNum, figsize=(5*(compNum), 6))

        # for i in range(compNum):
        #     axes[i].set_title(titles[i+1])
        #     axes[i].scatter(obsAround_imgs, obsAround_preds[i+1])
        
        # plt.savefig(f"{savepath}{os.sep}ObsAround_AllComparison.png")
        

