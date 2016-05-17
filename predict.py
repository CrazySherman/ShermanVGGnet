import caffe, cv2
import os, sys
## predict the whole shit
# image_dims is a fuqin resize flag, when don't need that shit cuz it uses skiamge package
# don't swap channel cuz cv2.imread exactly give u BGR format, which is demanded by CaffeNet


def predict(model_name, test_img_path, lim=None, batch=32):
    '''
        input: model_name -- caffemodel name under this dir
                test_img_path: the test image path
        output: iterable batch_size * 10 ndarray of probability predictions
    '''
    MODEL_FILE = 'deploy.prototxt'
    PRETRAINED = model_name
    count = 0
    imgs = []
    mean = [104, 117, 123]
    net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                           mean= np.array(mean).reshape(3,1,1),
                           raw_scale=255)
    for f in os.lstdir(test_img_path):
        f = os.path.join(test_img_path, f)
        # input dimension: 1 224 224
        img = cv2.imread(f)
        img = cv2.resize(img, (256,256))
        imgs.append(img)
        count += 1
        if count % batch == 0:
            yield net.predict(imgs)  # predict takes any number of images
            imgs = []
            # dummy test, only output the first batch result
            if lim:
                return
    if not imgs:
        yield net.predict(imgs) 

if __name__ == '__main__':

    if len(sys.argv) != 3:
        raise NameError('usage: predict <model name> <image dir>')
    # dummy test
    res = predict(sys.argv[1], sys.argv[2], lim=10)
    for r in res:
        print r