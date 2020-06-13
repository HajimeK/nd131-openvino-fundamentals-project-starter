https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/ssd

model

https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/ssd/model/ssd-10.onnx

sample

```
import numpy as np
from PIL import Image

def preprocess(img_path):
    input_shape = (1, 3, 1200, 1200)
    img = Image.open(img_path)
    img = img.resize((1200, 1200), Image.BILINEAR)
    img_data = np.array(img)
    img_data = np.transpose(img_data, [2, 0, 1])
    img_data = np.expand_dims(img_data, 0)
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[1]):
        norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data
```

Output of model

The model has 3 outputs. boxes: (1x'nbox'x4) labels: (1x'nbox') scores: (1x'nbox')
Dataset (Train and validation)

The SSD model was trained on 2017 COCO train data set - using mlperf/training/single_stage_detector repo , compute mAP on 2017 COCO val data set.