import pandas as pd

import caffe, cv2
import os, sys
import numpy as np
## predict the whole shit
# image_dims is a fuqin resize flag, when don't need that shit cuz it uses skiamge package
# don't swap channel cuz cv2.imread exactly give u BGR format, which is demanded by CaffeNet

h, w = 224, 224
def predict(model_name, test_img_path, lim=None, batch=32):
    '''
        input: model_name -- caffemodel name under this dir
                test_img_path: the test image path
        output: iterable batch_size * 10 ndarray of probability predictions
    '''
    caffe.set_mode_gpu()
    MODEL_FILE = 'deploy.prototxt'
    PRETRAINED = model_name
    count = 0
    imgs = []
    mean = [104, 117, 123]
    means = np.ones((3, h, w))
    means[0,:,:] *= mean[0]
    means[1,:,:] *= mean[1]
    means[2,:,:] *= mean[2]
    # dont use raw scale, cv2 input is perfect, dont modify that shit
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean=means)
    img_names = []
    for f in os.listdir(test_img_path):
        f_full = os.path.join(test_img_path, f)
        # input dimension: 1 224 224
        img = cv2.imread(f_full)
        img = cv2.resize(img, (256,256))
        imgs.append(img)
        count += 1
        img_names += [f]
        if count % batch == 0:
            yield net.predict(imgs, oversample=False), img_names  # predict takes any number of images
            imgs = []
            img_names = []
            # dummy test, only output the first batch result
            if lim:
                return
    if imgs:
        yield net.predict(imgs, oversample=False), img_names 

if __name__ == '__main__':

    if len(sys.argv) != 3:
        raise NameError('usage: predict <model name> <image dir>')
    # dummy test
    res = predict(sys.argv[1], sys.argv[2], lim=True)
    classes = [format("c%d" %x) for x in range(10)]
    
    names = []
    preds = []
    i = 0
    for r, n in res:
        names += n
        preds += [r]
        i +=r.shape[0]
        print ('Done %d' %i)
    
    print ("Done ")
    
    results = pd.DataFrame(np.concatenate(preds), columns=classes)
    #print (results.head(n=30))
    print results
    #results.loc[:,'img'] = pd.Series(names, index=results.index)
    #results.to_csv('vgg_submission.csv', index=False)
