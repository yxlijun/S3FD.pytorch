## S³FD: Single Shot Scale-invariant Face Detector ##
A PyTorch Implementation of Single Shot Scale-invariant Face Detector

### Description
Meanwhile train hand and head with S3FD,hand dataset is [Egohands Dataset](http://vision.soic.indiana.edu/projects/egohands/),head dataset is [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release),we can download [hand model](https://pan.baidu.com/s/1_d4HqioBJknGj2ypwtYaXg) and [face model](https://pan.baidu.com/s/1epyTAUc6qSt3oZ7veK4oEw)

### Requirement
* pytorch 0.3 
* opencv 
* numpy 
* easydict

### Prepare data 
1. download WIDER face dataset、Egohands dataset and SCUT-HEAD
2. modify data/config.py 
3. ``` python prepare_wider_data.py ```
4 ``` python prepare_handataset.py ```


### Train
We can choose different dataset to train different target[face,head,hand] 
``` 
python train.py --batch_size 4 --dataset face\hand\head
``` 

### Evalution
according to yourself dataset path,modify data/config.py 
1. Evaluate on AFW.
```
python afw_test.py
```
2. Evaluate on FDDB 
```
python fddb_test.py
```
3. Evaluate on PASCAL  face 
``` 
python pascal_test.py
```
4. test on WIDER FACE 
```
python wider_test.py
```
### Demo 
you can test yourself image
```
python demo.py
```

### Result
1. AFW PASCAL FDDB
<div align="center">
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/AFW.png" height="200px" alt="afw" >
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/pascal.png" height="200px" alt="pascal" >
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/FDDB.png" height="200px" alt="fddb" >     
</div>

	AFW AP=99.81 paper=99.85 
	PASCAL AP=98.77 paper=98.49
	FDDB AP=0.975 paper=0.983
	WIDER FACE:
	Easy AP=0.925 paper = 0.927
	Medium AP=0.925 paper = 0.924
	Hard AP=0.854 paper = 0.852

2. demo
<div align="center">
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/tmp/test2.jpg" height="400px" alt="afw" >
</div>


### References
* [S³FD: Single Shot Scale-invariant Face Detector](https://arxiv.org/abs/1708.05237)
* [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)