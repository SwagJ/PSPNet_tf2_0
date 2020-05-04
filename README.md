# PSPNet_tensorflow

## Introduction
  This is a repository forked from [hellochick/PSPNet-tensorflow](https://github.com/hellochick/PSPNet-tensorflow). The 
  code has been updated to TensorFlow 2.1 and adapted for our project.

## Inference

### Inference on datasets
To do semantic labeling on dataset `MS-COCO 2014` or `MegaDepth`, use the following command:
```
python inference_dataset.py --data_path=DATA_PATH --dataset=coco  
```

Options:
```
--data_path: DATA_PATH set in SuperPoint
--dataset: coco or megadepth
--flipped-eval 
```
