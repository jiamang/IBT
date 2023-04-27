# Exploiting Inductive Bias in Transformer for Point Cloud  Classification and Segmentation
This repository contains PyTorch implementation for IBT : **Exploiting Inductive Bias in Transformer for Point Cloud  Classification and Segmentation.** 
Our code skeleton is borrowed from antao97/dgcnn.pytorch

Our IBT module is as follows:
![image](https://github.com/jiamang/IBT/blob/main/image/IBT.png)


## Requirements
Python >= 3.7 , PyTorch >= 1.2  , CUDA >= 10.0  , Package: glob, h5py, sklearn, plyfile, torch_scatter


## Point Cloud Classification
Note: You can choose to implement the classification experiment on the ModelNet40 or ScanObjectNN dataset, just select the corresponding data loading function.
### Run the training script:
``` 
python main_cls.py --exp_name=cls_1024_scan --num_points=1024 --k=40 
```
### Run the evaluation script after training finished:
``` 
python main_cls.py --exp_name=cls_1024_eval_scan --num_points=1024 --k=40 --eval=True --model_path=outputs/cls_1024_scan/models/model.t7
```

## Point Cloud Part Segmentation
Note: There are two options for training on the full dataset and for each class individually. In order to obtain more detailed features, we set the k value to 80 for Part Segmentation.
### Run the training script:
· Full dataset
```
python main_partseg.py --exp_name=partseg 
```
· With class choice, for example airplane
```
python main_partseg.py --exp_name=partseg_airplane --class_choice=airplane
```
### Run the evaluation script after training finished:
· Full dataset
```
python main_partseg.py --exp_name=partseg_eval --eval=True --model_path=outputs/partseg/models/model.t7
```
· With class choice, for example airplane
```
python main_partseg.py --exp_name=partseg_airplane_eval --class_choice=airplane --eval=True --model_path=outputs/partseg_airplane/models/model.t7
```
