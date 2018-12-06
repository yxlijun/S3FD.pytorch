## S³FD: Single Shot Scale-invariant Face Detector ##
A PyTorch Implementation of Single Shot Scale-invariant Face Detector.

### Requirement
* pytorch 0.3 
* opencv 
* numpy 
* easydict

### P	repare WIDER data 
1. download wider face dataset 
2. modify data/config.py wider face path
3. ``` python prepare_wider_data.py ```

### Train 
``` 
python train.py --batch_size 4 --lr 1e-4
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
<div align="center">
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/AFW.png" height="300px" alt="afw" >
<img src="https://github.com/yxlijun/S3FD.pytorch/blob/master/img/pascal.png" height="300px" alt="pascal" >    
</div>

1. AFW AP=99.43 paper=99.85
![](https://github.com/yxlijun/S3FD.pytorch/blob/master/img/AFW.png)
2. PASCAL AP=98.77 paper=98.49
![](https://github.com/yxlijun/S3FD.pytorch/blob/master/img/pascal.png)
3. FDDB AP=0.969 paper=0.983
![](https://github.com/yxlijun/S3FD.pytorch/blob/master/img/FDDB.png)
4. test demo
![](https://github.com/yxlijun/S3FD.pytorch/blob/master/tmp/test2.jpg)

### References
