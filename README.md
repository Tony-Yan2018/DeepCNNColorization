# DeepCNNColorization for B&W images/videos
## Basic model
This CNN model is inspired from the [lizuka et al] paper:[Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultanenous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/data/colorization_sig2016.pdf)  
The implementation is Tensorflow and Keras.  
Due to very limited hardware power(MX150 2GB), some minor modifications are made in order to train the model locally.
## First success
The first successful model is shown as the following:![firstSuccessModel](https://github.com/Tony-Yan2018/DeepCNNColorization/blob/master/README_im/firstSuccessModel.png)  

Training details of the first success model:  
Epochs: 10  
BatchSize: 8  
Total sample number: 36600  
Accuracy: 0.6782  
Loss: 7.1268e-3  
Training time: at least more than 10 hours  

Some of the best colorizations made from the first successful model:  
![firstSuccess1](https://github.com/Tony-Yan2018/DeepCNNColorization/blob/master/README_im/firstSuccess1.png)![firstSuccess2](https://github.com/Tony-Yan2018/DeepCNNColorization/blob/master/README_im/firstSuccess2.png)![firstSuccess3](https://github.com/Tony-Yan2018/DeepCNNColorization/blob/master/README_im/firstSuccess3.png)  
![firstSuccess4](https://github.com/Tony-Yan2018/DeepCNNColorization/blob/master/README_im/firstSuccess4.png)![firstSuccess5](https://github.com/Tony-Yan2018/DeepCNNColorization/blob/master/README_im/firstSuccess5.png)![firstSuccess6](https://github.com/Tony-Yan2018/DeepCNNColorization/blob/master/README_im/firstSuccess6.png)





