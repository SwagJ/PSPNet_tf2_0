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

### Inference on single image
To get result on single images, use the following command:
```
python inference.py --img-path=./input/test.png --dataset cityscapes  
```
Options:
```
--dataset cityscapes or ade20k
--flipped-eval 
--checkpoints /PATH/TO/CHECKPOINT_DIR
```
## Checkpoints
Checkpoints for ade20k has been included inside the `model` directory. To get checkpoints trained on `cityscapes`
please go to [Google Drive](https://drive.google.com/drive/folders/1S90PWzXEX_GNzulG1f2eTHvsruITgqsm?usp=sharing). Note: Select the checkpoint corresponding to the dataset.
