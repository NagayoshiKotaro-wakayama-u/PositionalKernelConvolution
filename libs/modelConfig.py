from argparse import ArgumentParser
from libs.models import branchPKConv_lSite,PKConvUNetAndSiteCNN,PConvUnetModel,chConcatPConvUNet,PConvLearnSite,PKConvLearnSite,PConv_ConditionalSite,branchPConv_lSite,sharePConv_lSite
import os
import pdb

def parse_args(isTrain=True):
    if isTrain:
        descript = 'Training script for PConv inpainting'
    else:
        descript = 'Test script for PConv inpainting'
    
    parser = ArgumentParser(description=descript)
    #========================================
    # trainとtest共通部分
    parser.add_argument('experiment',type=str,help='name of experiment, e.g. \'normal_PConv\'')
    parser.add_argument('-dataset','--dataset',type=str,default='stripe-rectData',help='name of dataset directory (default=gaussianToyData)')
    parser.add_argument('-imgw','--imgw',type=int,default=512,help='input width')
    parser.add_argument('-imgh','--imgh',type=int,default=512,help='input height')
    parser.add_argument('-lr','--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('-pt','--pretrainModel',type=str,default="")
    
    #===============
    ## 提案手法
    parser.add_argument('-pk','--PKConv',action='store_true')
    parser.add_argument('-ope','--operation',type=str,default='mul',help='mul or add (default=mul)')
    parser.add_argument('-concatPConv','--concatPConv',action='store_true')
    parser.add_argument('-siteCNNsinglePath','--siteCNNsinglePath',action='store_true')
    parser.add_argument('-useSiteGAP','--useSiteGAP',action='store_true')
    parser.add_argument('-learnSitePConv','--learnSitePConv',action='store_true')
    parser.add_argument('-siteLayers','--siteLayers',type=lambda x:list(map(int,x.split(","))),default="1")
    parser.add_argument('-learnSitePKConv','--learnSitePKConv',action='store_true')
    parser.add_argument('-obsOnlyL1','--obsOnlyL1',action='store_true')
    parser.add_argument('-fullLayer','--learnSiteFullLayer',action='store_true')
    parser.add_argument('-phase','--phase',type=int,default=0)
    parser.add_argument('-nonNegSConv','--nonNegSConv',action='store_true')

    ## 位置特性関連
    parser.add_argument('-site','--sitePath',type=lambda x:list(map(str,x.split(","))),default="upperDepth_layer3.png")
    parser.add_argument('-smax','--siteMax',type=float,default=1,help="位置特性の最大値")
    parser.add_argument('-smin','--siteMin',type=float,default=0.1,help="位置特性の最小値")
    parser.add_argument('-siteFs','--siteConvFs',type=lambda x:list(map(int,x.split(","))),default="1,1,1,1",help="list of int (channel of site-Convolution)")
    parser.add_argument('-siteSig','--siteSigmoid',action='store_true')
    parser.add_argument('-siteClip','--siteClip',action='store_true')
    parser.add_argument('-Lasso','--Lasso',action='store_true')
    parser.add_argument('-localLasso','--localLasso',action='store_true')
    parser.add_argument('-learnMultiSiteW','--learnMultiSiteW',action='store_true')
    parser.add_argument('-cPConv','--cPConv',action='store_true')# コンディショナルなPConv（震源別など）
    parser.add_argument('-branchlSite','--branchLearnSitePConv',action='store_true') # 枝分かれした位置特性CNNを持つ
    parser.add_argument('-branchlSitePKConv','--branchLearnSitePKConv',action='store_true') # 枝分かれ（多チャンネル対応）＆PKConv
    parser.add_argument('-sharePConv','--sharePConv_lSite',action='store_true')
    parser.add_argument('-classNum','--classNum',type=int,default=3)
    #===============
    #========================================

    if isTrain:
        # train時の設定
        parser.add_argument('-existOnly','--existOnly',action='store_true')
        parser.add_argument('-lossw','--lossWeights',type=lambda x:list(map(float,x.split(","))),default="1,6,0.1")
        parser.add_argument('-epochs','--epochs',type=int,default=400,help='training epoch')
        parser.add_argument('-sCNNpretrain','--sCNNpretrain',type=int,default=0)
        parser.add_argument('-patience','--patience',type=int,default=10)
        parser.add_argument('-switchPhase','--switchingTrainPhase',action='store_true')
        parser.add_argument('-earlySiteStop','--earlySiteFeatureStop',action='store_true')
    else:
        # test時の設定
        parser.add_argument('-plotTest','--plotTest',action='store_true',help="テスト結果そのものを画像として保存するかどうか")
        parser.add_argument('-plotComp','--plotComparison',action='store_true',help="分析結果をプロットするか")
        parser.add_argument('-flatSite','--flatSiteFeature',action='store_true')
        parser.add_argument('-siteonly','--siteonly',action='store_true')

    return  parser.parse_args()

def modelBuild(modelType,argsObj,isTrain=True):
    # pdb.set_trace()
    existPath = f"data{os.sep}sea.png" if "quake" in argsObj.dataset else ""
    keyArgs = {"img_rows":argsObj.imgh,"img_cols":argsObj.imgw,"exist_point_file":existPath}
    if isTrain:
        keyArgs.update({"existOnly":argsObj.existOnly})

    if modelType=="pconv":
        ## 既存法
        return PConvUnetModel(**keyArgs)
    elif modelType=="pkconvAndSiteCNN":
        ## 提案法
        keyArgs.update({
                "siteConvF":argsObj.siteConvFs,
                "siteInputChan":len(argsObj.sitePath),
                "site_range":[argsObj.siteMax,argsObj.siteMin],
                "siteSigmoid":argsObj.siteSigmoid,
                "siteSinglePath":argsObj.siteCNNsinglePath,
                "siteClip":argsObj.siteClip,
                "learnMultiSiteW":argsObj.learnMultiSiteW,
                "useSiteGAP":argsObj.useSiteGAP,
                "opeType":argsObj.operation,
            })
        if isTrain:
            keyArgs.update({
                "SCNN_pretrain":argsObj.sCNNpretrain    
            })
        return PKConvUNetAndSiteCNN(**keyArgs)
    
    elif modelType=="conc-pconv":
        # 位置特性を結合して入力する
        return chConcatPConvUNet(**keyArgs)

    elif modelType=="learnSitePKConv":
        # 学習可能な位置特性を持ったPKConv
        keyArgs.update({
                "opeType":argsObj.operation,
                "obsOnlyL1":argsObj.obsOnlyL1
            })
        return PKConvLearnSite(**keyArgs)
    elif modelType=="conditionalPConv":
        keyArgs.update({
            "siteNum":argsObj.classNum,
            "opeType":argsObj.operation,
            "obsOnlyL1":argsObj.obsOnlyL1
        })
        return PConv_ConditionalSite(**keyArgs)
    elif modelType=="branch_lSitePKConv":
        keyArgs.update({
            "nonNeg":argsObj.nonNegSConv,
            "opeType":argsObj.operation,
            "siteLayers":argsObj.siteLayers,
        })
        return branchPKConv_lSite(**keyArgs)
    else:
        keyArgs.update({
            "encStride":(1,1),
            "opeType":argsObj.operation,
            "obsOnlyL1":argsObj.obsOnlyL1,
            "siteLayers":argsObj.siteLayers,
            "fullLayer":argsObj.learnSiteFullLayer,
            "localLasso":argsObj.localLasso,
            "Lasso":argsObj.Lasso,
        })

        if modelType=="branch_lSitePConv":
            keyArgs.update({"nonNeg":argsObj.nonNegSConv})
            return branchPConv_lSite(**keyArgs)
        elif modelType=="sharePConv_lSite":
            return sharePConv_lSite(**keyArgs)
        elif modelType=="learnSitePConv":
            # 学習可能な位置特性を持ったPConv
            return PConvLearnSite(**keyArgs)

        return None
