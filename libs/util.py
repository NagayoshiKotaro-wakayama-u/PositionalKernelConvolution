import numpy as np
from PIL import Image
import pickle
import pdb
import matplotlib
import matplotlib.pyplot as plt
import os
from numpy.lib.stride_tricks import as_strided


def rangeError(pre,tru,domain=[-1.0,0.0],opt="PSNR"): # 欠損部含めた誤差 pred:予測値, true:真値 , domain:値域(domain[0]< y <=domain[1])
    # ある値域の真値のみで誤差を測る
    domain = [domain[0],domain[1]] # normalyse
    inds = np.where(np.logical_and(tru>domain[0],tru<=domain[1]))
    if inds[0].shape[0]==0: # 値がない場合はNaN
        return np.NaN

    # pdb.set_trace()
    p = pre[inds[0],inds[1]] if len(pre.shape) == 2 else pre[inds[0]]
    t = tru[inds[0],inds[1]] if len(tru.shape) == 2 else tru[inds[0]]

    if opt=="PSNR":
        error_ = PSNR(p,t)
        return error_

    error_ = t-p
    
    if opt=="MA": # MAE
        error_ = np.mean(np.abs(error_))
    elif opt=="MS": # MSE
        error_ = np.mean(error_**2)
    elif opt=="A":
        error_ = np.abs(error_)

    return error_

def loadSiteImage(paths, smax=1.0, smin=0.0):#paths=[str1,str2,...,strN]
    _images = []
    for path in paths:
        _img = np.array(Image.open(f"{path}"))[:,:,np.newaxis]
        
        immax = np.max(_img)
        immin = np.min(_img)
        imrange = immax-immin

        # 線形に正規化
        _img = (_img-immin)*(smax-smin)/imrange + smin

        _images.append(_img)
    # 
    if len(_images)>1:
        _images = np.concatenate(_images,axis=2)[np.newaxis,:,:,:]
    else:
        _images = np.array(_images)

    return _images

def PSNR(y_pred,y_true,maxI=1,exist=None):
    if exist is not None:
        # pdb.set_trace()
        y_pred = y_pred[exist==1]
        y_true = y_true[exist==1]
    mse = np.mean(np.square(y_pred - y_true))
    return 10.0 * np.log( maxI**2 / mse) / np.log(10.0)

def pickleOpen(path):
    f = open(path,"rb")
    res = pickle.load(f)
    f.close()
    return res

# 1 or 2次元データ（x）中の上位何割か（ratio）の座標(index)を取得
def extractFromTop(x,ratio):
    x = np.squeeze(x)
    shape = x.shape
    length = np.prod(shape)

    # 2次元の場合は1次元にする
    if len(shape) == 2:
        x = np.reshape(x,length)
    elif len(shape) > 2:
        raise ValueError("dimension of shape must be 1 or 2")

    num = int(ratio*length)
    topInds = np.argsort(x)[-num:] # 昇順に並び替えて後ろから取得

    if len(shape) == 2:
        row,col = shape
        resultInds = [[ind//row, ind%col] for ind in topInds]
    else:
        resultInds = topInds

    return resultInds



def pool2d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')
    # pdb.set_trace()

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])

    A_w = as_strided(A, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return A_w.mean(axis=(2, 3))


# 複数のデータを一つのプロットに
def plotDatas(datas, title="", savepath="", xlabel="epoch", ylabel="", labels=None, style=None):
    plt.clf()
    plt.close()
    plt.title(title)
    plt.xlabel(xlabel)
    if ylabel=="":
        ylabel = title
    plt.ylabel(ylabel)

    if not isinstance(datas,list):
        datas = [datas]

    for i,data in enumerate(datas):
        
        # x軸はエポック
        ## listかnp.arrayを使用可能
        if isinstance(data,list):
            epochs = range(len(data))
        elif isinstance(data,np.array):
            epochs = range(data.shape[0])
        else:
            return

        args = [epochs, data]
        kwargs = {}

        if labels is not None:
            kwargs.update({"label":labels[i]})
        
        # リストでもリストじゃなくても可
        if style is not None:
            if isinstance(style,list) and len(style) > 1:
                args.append(style[i])
            else:
                args.append(style)

        plt.plot(*args,**kwargs)

    
    if labels is not None:
        plt.legend()

    if savepath != "":
        plt.savefig(savepath)
        plt.clf()
        plt.close()

    return

def plotImgs(imgs,savename,exist=None,titles=["masked","inpainted","original"]):

    num = len(imgs)
    plt.clf()
    plt.close()
    _, axes = plt.subplots(1,num, figsize=(5*num,5))
    for i in range(num):
        axes[i].set_title(titles[i])
        axes[i].imshow(imgs[i])
    
    plt.savefig(savename)
    return


def nonhole(x,hole): # 欠損部以外の値を取り出す
    shape = x.shape
    flatt = np.reshape(x,(np.product(shape)))
    holes = np.reshape(hole,(np.product(shape)))
    tmp = []
    for pix,hole in zip(flatt,holes):
        if np.sum(hole) < 1e-10:
            continue
        tmp.append(pix)

    return np.array(tmp)



def cmap(x,exist_rgb=None,sta=[222,222,222],end=[255,0,0]): #x:gray-image([w,h]) , sta,end:[B,G,R]
    vec = np.array(end) - np.array(sta)
    res = []
    for i in range(x.shape[0]):
        tmp = []
        for j in range(x.shape[1]):
            tmp.append(np.array(sta)+x[i,j]*vec)
        res.append(tmp)
    res = np.array(res).astype("uint8")
    if exist_rgb is not None:
        if len(exist_rgb.shape)==2:
            exist_rgb = np.tile(exist_rgb[:,:,np.newaxis],[1,1,3])
        res[exist_rgb==0] = 255

    return res

def clip(x,sta=-0.1,end=0.1): # Clip the value
    x[x<sta] = sta
    x[x>end] = end
    dist = end-sta
    res = (x-sta)/dist
    return res


class resultInpaint():
    # imgs:真値　mask:マスク画像　exist:予測領域画像　experiment_path:予測結果のpickleが入っているディレクトリ
    def __init__(self, imgs, mask, exist, experiment_path, regions={}) -> None:
        # maskとexistは0～1
        assert (np.max(mask) == 1 and np.min(mask) == 0) and (np.max(exist) == 1 and np.min(exist) == 0)
                
        #===============
        # path関連
        self.experimentPath = experiment_path
        self.resultPath = f"{self.experimentPath}{os.sep}result"
        self.compPath = f"{self.resultPath}{os.sep}comparison"
        self.regionPath = f"{self.resultPath}{os.sep}region"
        for DIR in [self.resultPath, self.compPath, self.regionPath]:
            if not os.path.isdir(DIR):
                os.makedirs(DIR)
        #===============
        # データ関連
        self.imgs = np.squeeze(imgs)
        self.mask = np.squeeze(mask)
        # pdb.set_trace()
        self.preds = pickle.load(open(f"{self.resultPath}{os.sep}testSummary.pickle","rb"))["preds"]
        self.preds = np.squeeze(np.array(self.preds))
        self.exist = exist

        self.shape = self.imgs.shape[1:]

        #===============
        # plot関連
        self.cm_bwr = plt.get_cmap("bwr") # カラーマップ

        #===============
        # region関連
        self.regions = {}
        for key in regions.keys():
            self.addRegion(key,regions[key]) # 座標の大小関係をチェックしながら追加
        self.regPSNRs = {}
        #===============

    def plot3x3Analyse(self,mask,pred,img,exist,savepath="comparison.png"):# 3x3プロットを保存
        self.plotClear()# pltをクリア
        mask_rgb = np.tile(mask[:,:,np.newaxis],[1,1,3])
        exist_rgb = np.tile(exist[:,:,np.newaxis],[1,1,3])

        _, axes = plt.subplots(3, 3, figsize=(16, 15))
        ### 入力・予測・真値の比較
        titles = ["input","pred","original"]
        masked = mask*img
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
        err = pred - img
        err = self.cm_bwr(clip(err,-0.1,0.1))[:,:,:3] # -0.1~0.1の偏差をカラーに変換
        axes[2,1].imshow(err*exist_rgb,vmin=0,vmax=1.0)

        plt.savefig(savepath)

    def plotRegion3x3Analyse(self,regionName):# ある地域の3x3プロットを保存
        # クロップ
        imgs,mask,preds,exist = self.cropRegion(regionName)
        self.plotClear()
        ite = 0

        # regionのディレクトリ下に新たにディレクトリを追加
        regPath = f"{self.regionPath}{os.sep}{regionName}"
        if not os.path.isdir(regPath):
            os.makedirs(regPath)

        # 全データについてプロット
        for img,pred in zip(imgs,preds):
            print(f"\rplot {regionName}... ",end="")
            savepath = f"{regPath}{os.sep}{regionName}{ite}.png"
            # 3x3のプロット
            self.plot3x3Analyse(mask,pred,img,exist,savepath=savepath)
            ite += 1
        
        return

    def allRegionPSNR(self):# 登録されている領域のPSNRを計算して表示
        returndict = {}
        for key in self.regions.keys():
            # PSNRを計算（すでに計算済みの場合はregPSNRsから取得）
            psnr = self.PSNRs[key] if key in self.regPSNRs.keys() else self.regionPSNR(key)
            
            print(f"{key}:PSNR={psnr}")
            returndict.update({key:psnr})
        return returndict
    
    def regionPSNR(self,regionName):# ある領域のPSNRを全データで平均して返す
        imgs,_,preds,exist = self.cropRegion(regionName)
        # psnr = PSNR(preds, imgs)
        psnr = PSNR(preds, imgs, exist=np.tile(exist[np.newaxis],[self.imgs.shape[0],1,1]))
        # psnr = self.PSNR(preds, imgs, exist)
        # self.regPSNRs.update({regionName:psnr})
        return psnr

    def addRegion(self,regionName,region):# region=[0,0,512,512]など
        # regionの大小関係
        assert (region[0] < region[2]) and (region[1] < region[3])
        self.regions.update({regionName:region})
        return

    def allPSNR(self):
        # PSNRを計算（すでに計算済みの場合はregPSNRsから取得）
        # psnr = PSNR(self.preds, self.imgs, exist=)
        psnr = PSNR(self.preds, self.imgs, exist=np.tile(self.exist[np.newaxis],[self.imgs.shape[0],1,1]))
        # psnr = self.PSNR(self.preds, self.imgs, exist=self.exist)
        print(f"{self.experimentPath}:PSNR={psnr}")
        return psnr

    def cropRegion(self,regionName):# regionを切り取る
        region = self.regions[regionName]
        imgs = self.imgs[:, region[0]:region[2], region[1]:region[3]]
        mask = self.mask[region[0]:region[2], region[1]:region[3]]
        preds = self.preds[:, region[0]:region[2], region[1]:region[3]]
        exist = self.exist[region[0]:region[2], region[1]:region[3]]
        return imgs,mask,preds,exist

    def plotClear(self):# pltをクリア
        plt.clf()
        plt.close()
        return

    def PSNR(self,preds,imgs,exist):
        psnrs = []
        for img,pred in zip(imgs,preds):
            psnrs.append(PSNR(pred, img, exist=exist))
        
        return np.mean(psnrs)


class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)