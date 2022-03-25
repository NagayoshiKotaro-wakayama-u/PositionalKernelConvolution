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

    # trainとtest共通部分
    #========================================
    parser.add_argument('experiment',type=str,help='name of experiment, e.g. \'normal_PConv\'')
    parser.add_argument('-dataset','--dataset',type=str,default='stripe-rectData',help='name of dataset directory (default=gaussianToyData)')
    parser.add_argument('-imgw','--imgw',type=int,default=512,help='input width')
    parser.add_argument('-imgh','--imgh',type=int,default=512,help='input height')
    parser.add_argument('-lr','--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('-pt','--pretrainModel',type=str,default="")
    
    #===============
    ## 提案手法
    parser.add_argument('-ope','--operation',type=str,default='mul',help='mul or add (default=mul)')

    parser.add_argument('-learnSitePConv','--learnSitePConv',action='store_true')
    parser.add_argument('-learnSitePKConv','--learnSitePKConv',action='store_true')
    parser.add_argument('-branchlSitePConv','--branchLearnSitePConv',action='store_true') # 枝分かれした位置特性CNNを持つ
    parser.add_argument('-branchlSitePKConv','--branchLearnSitePKConv',action='store_true') # 枝分かれ（多チャンネル対応）＆PKConv

    parser.add_argument('-siteLayers','--siteLayers',type=lambda x:list(map(int,x.split(","))),default="1")
    parser.add_argument('-phase','--phase',type=int,default=0)
    parser.add_argument('-nonNegSConv','--nonNegSConv',action='store_true')
    #===============
    #========================================

    if isTrain:
        # train時の設定
        parser.add_argument('-existOnly','--existOnly',action='store_true')
        parser.add_argument('-lossw','--lossWeights',type=lambda x:list(map(float,x.split(","))),default="1,6,0.1")
        parser.add_argument('-epochs','--epochs',type=int,default=400,help='training epoch')
        parser.add_argument('-patience','--patience',type=int,default=10)
        parser.add_argument('-switchPhase','--switchingTrainPhase',action='store_true', help="学習中にフリーズする箇所を切り替える")
    else:
        # test時の設定
        parser.add_argument('-plotTest','--plotTest',action='store_true',help="テスト結果そのものを画像として保存するかどうか")
        parser.add_argument('-plotComp','--plotComparison',action='store_true',help="分析結果をプロットするか")
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

    elif modelType=="learnSitePKConv": # 提案手法
        # 学習可能な位置特性を持ったPKConv
        keyArgs.update({
                "opeType":argsObj.operation,
                "obsOnlyL1":argsObj.obsOnlyL1
            })
        return PKConvLearnSite(**keyArgs)
    
    elif modelType=="branch_lSitePKConv": # 提案手法
        
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
            "siteLayers":argsObj.siteLayers
        })

        if modelType=="branch_lSitePConv": # 提案手法
            keyArgs.update({
                "nonNeg":argsObj.nonNegSConv,
            })
            return branchPConv_lSite(**keyArgs)

        elif modelType=="learnSitePConv": # 提案手法
            return PConvLearnSite(**keyArgs)

        return None
