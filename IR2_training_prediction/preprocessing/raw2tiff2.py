import numpy as np
import os,glob
import shutil
from tifffile import imsave, imread
from PyQt5.QtWidgets import QFileDialog, QApplication

def get_meta(paramFile):
    param_list={}
    with open(paramFile) as f:
        lines = f.readlines()
        for l in lines:
            try:
                (val,p)=l.split(":")
                try:
                    # save value as an integer
                    param_list[p.strip()]=np.float(val)
                except ValueError:
                    # otherwise save as text with \n and \t stripped
                    param_list[p.strip()]=val.strip()
            except ValueError:
                pass
    return (param_list)

def get_img_shape(meta):
    shape = (int(meta['Planes']),
                int(meta['ROIHeight']),
                int(meta['ROIWidth']))
    return shape

def load_images(fList,shape,offset=4,delta=4):
    ''' Load raw data as nD numpy array.
    '''
    print('Files detected: %02d'%len(fList))
    ext = os.path.splitext(fList[0])[-1]
    print('Files format:   '+ext+', loading data...')
    if ext=='.raw':
        imgs = np.zeros((len(fList),*shape)).astype(np.uint16)
        for i in range(len(fList)):
            with open(fList[i],'rb') as fn:
                tmp = np.fromfile(fn,dtype=np.uint16)
                # tmp = np.clip(tmp,0,2**16-1).astype(np.uint16)
                tmp = np.stack([ int(offset/2)+tmp[(np.prod(shape[1:])+int(delta/2))*i:(np.prod(shape[1:])+int(delta/2))*(i+1)] for i in range(shape[0])])
                imgs[i] = np.stack([ j[2:].reshape(shape[1:]) for j in tmp ])
        del tmp
    elif (ext=='.tif')or(ext=='.tiff'):
        imgs = np.stack( [ imread(i) for i in fList ] )
    if len(imgs.shape)==4:
        axID = 'CZYX'
    elif len(imgs.shape)==3:
        axID = 'CXY'
    return imgs.astype(np.uint16), axID

def load_raw_images(files_raw,img_shape):
    (imgs, _) = load_images(files_raw,img_shape)
    return imgs

def convert2tiff(inpath, paramFile, flist, outpath, fuse=False):
    meta = get_meta(paramFile)
    img_shape = get_img_shape(meta)
    print(paramFile)
    # for f in flist:
    #     print(f)
    print('Shape: ',img_shape)

    imgs = load_raw_images(flist,img_shape)
    for img, f in zip(imgs,flist):
        #fix saturation point
        img[img==1] = 2**16-1
        print('Processing: ',f)
        fname = os.path.splitext(os.path.split(f)[-1])[0]
        fname = fname.replace('channel=','channel=ch')
        idx = fname.find('channel=ch')+len('channel=ch')
        chname = '%02d'%(int(fname[idx:idx+2])-1)
        fname = fname[:idx]+chname+fname[idx+2:]
        imsave(os.path.join(outpath,fname+'.tif'),img)
        #os.remove(f)
    newParamName = fname.replace('channel=ch01','channel=ch[CCC]')+'_params.txt'
    shutil.copy(paramFile,os.path.join(outpath,newParamName))

def fuse_illuminations(imgName_R,imgName_L,x_0=1024):
    img_R = imread(imgName_R)
    img_L = imread(imgName_L)
    (shapeZ,shapeX,shapeY) = img_L.shape

    sigmoid_steepness=0.05
    x_values=np.linspace(1,shapeX,shapeX)
    y_values=np.linspace(1,shapeY,shapeY)
    X,Y=np.meshgrid(x_values,y_values)

    sigmoid_L=1-1/(1+np.exp(-sigmoid_steepness*(X-x_0)))
    sigmoid_R=1/(1+np.exp(-sigmoid_steepness*(X-x_0)))

    fused_stack = np.zeros(img_L.shape).astype(np.float)
    for m in range(0, shapeZ):
        fused_stack[m,:,:]=img_L[m,:,:]*sigmoid_L+img_R[m,:,:]*sigmoid_R
    fused_stack = np.clip(fused_stack,0,2**16-1).astype(np.uint16)
    imsave(imgName_L.replace(',_direction=02',''),fused_stack)
    return

if __name__ == "__main__":
    app = QApplication([])
    # path = 'kdrlGFP-10dpf-AF800-AF647-fish1,_1-31-2020_2-53-53_PM'
    path = QFileDialog.getExistingDirectory(
                None, 'Select a directory', 
                os.path.join('Users','nicol','OneDrive','Desktop','test_kdrlGFP_AF647AF800'))
    drive, path = os.path.splitdrive(path)
    path = os.path.join(drive,os.sep,path)
    paramFile = os.path.join(path,'Experimental Parameters.txt')
    flist = glob.glob(os.path.join(path,'*.raw'))
    flist.sort()
    convert2tiff(path, paramFile, flist, path)

