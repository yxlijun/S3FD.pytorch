## S³FD: Single Shot Scale-invariant Face Detector ##
A PyTorch Implementation of Single Shot Scale-invariant Face Detector.

### requirement
* pytorch 0.3 
* opencv 
* numpy 
* easydict

### prepare wider dataset 
1. download wider face dataset 
2. modify data/config.py wider face path
3. ``` python prepare_wider_data.py ```

### train 
``` 
python train.py --batch_size 4 \\
				--lr 1e-4
``` 

### test 
	according to yourserlf dataset path,modify data/config.py 
1. test on AFW
```
	python afw_test.py
```

2. test on fddb 
```
   python fddb_test.py
```
3. test on pascal face 
``` 
	python pascal_test.py
```
4. test on wider val 
```
	python wider_test.py
```

### demo 
```
	python demo.py
```

## result
1. afw
![](https://github.com/yxlijun/S3FD.pytorch/blob/master/img/AFW.png)
2. pascal 
![](https://github.com/yxlijun/S3FD.pytorch/blob/master/img/pascal.png)
3. demo 
![](https://github.com/yxlijun/S3FD.pytorch/blob/master/tmp/test2.jpg)

