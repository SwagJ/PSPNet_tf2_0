# PSPNet_tensorflow

## Introduction
  This is a repository forked from [hellochick/PSPNet-tensorflow](https://github.com/hellochick/PSPNet-tensorflow). The 
  code has been updated to TensorFlow 2.1 and adapted for our project. Moreover, eager execution is supported.

## Install
   - Get restore checkpoint from [Google Drive](https://drive.google.com/drive/folders/1S90PWzXEX_GNzulG1f2eTHvsruITgqsm) 
   and put into `checkpoint` directory. Note: Select the checkpoint corresponding to the dataset.
   - Get the `.npy` format checkpoint converted from the above `.ckpt` files from 
   [Google Drive](https://drive.google.com/file/d/1KHp-41Y50RJlv1hQ1TpLbio0uWjIwc-z/view?usp=sharing) 
   and put directly under the project directory. Note: `.npy` file is created by `ckpt2npy.py`.

## Inference

### Eager mode
   To get result on single images, use the following command:
   ```
   python inference_eager.py --img-path=./input/test1.png  
   ```
    
   Options:
   ```
   --checkpoints: path to checkpoint file in .npy format, default ./checkpoint.npy
   --flipped-eval
   --save-dir: directory to save result, default ./output/
   ```
### Graph mode
#### Inference on single image
   ```
   python inference_graph.py --img-path=./input/test1.npg --dataset=ade20k  
   ```
   Options:
   ```
   --checkpoints: path to checkpoint directory in .ckpt format, default ./checkpoint
   --flipped-eval
   --save-dir: directory to save result, default ./output/
   ```
#### Inference on datasets
   To do semantic labeling on the whole dataset `MS-COCO 2014` or `MegaDepth`, use the following command:
   ```
   python inference_graph_dataset.py --data_path=DATA_PATH --dataset=coco  
   ``` 
   where data_path is the same DATA_PATH set in SuperPoint, dataset can be chosen between `coco` and `megadepth`.
   Options:
   ```
   --flipped-eval 
   ```
