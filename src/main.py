# coding: utf-8  
import network
import network2
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU
import mnist_loader
import overfitting2
import json

if __name__ == '__main__':
    #training_data,validation_data,test_data=mnist_loader.load_data_wrapper()
    #print len(training_data)
    #print len(validation_data)
    #print len(test_data)

    #network:二次代价函数
    #net = network.Network([784,30,10])
    #net.SGD(training_data,30,10,3.0,test_data=test_data)
    
    #network2:交叉熵代价函数
    #net = network2.Network([784,30,30,30,10])
    #net = network2.Network([784,30,10])
    #net.large_weight_initializer()
    #net.SGD(training_data,30,10,0.1,evaluation_data=validation_data,monitor_evaluation_accuracy=True)

    #L2规范化:初始化权重未优化
    #net = network2.Network([784,30,10], cost=network2.CrossEntropyCost)
    #net.large_weight_initializer()
    #net.SGD(training_data[:1000],400,10,0.5,evaluation_data=test_data,lmbda=0.1,monitor_evaluation_cost=True,monitor_evaluation_accuracy=True,monitor_training_cost=True,monitor_training_accuracy=True)

    #L2规范化:初始化权重已优化
    """
    flag  = int(raw_input("if training(1) or use the exists data(0)?"))
    filename = "out_network2"
    if flag:
        net = network2.Network([784,30,10], cost=network2.CrossEntropyCost)
        evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data,30,10,0.5,lmbda=5.0,\
                    evaluation_data=validation_data,\
                    monitor_evaluation_cost=True,\
                    monitor_evaluation_accuracy=True,\
                    monitor_training_cost=True,\
                    monitor_training_accuracy=True)
        f = open(filename,"w")
        json.dump([evaluation_cost, evaluation_accuracy,training_cost,training_accuracy],f)
        f.close()
    overfitting2.make_plots(filename, 30, 0, 0, 0, 0, 50000)
    """
    
    #卷积网络开始
    training_data,validation_data,test_data=network3.load_data_shared()
    mini_batch_size=10
    """
    #浅层网络(规范化、柔性最大值作为最终层):作为开始和基线，以和卷积网络对比
    net = Network([FullyConnectedLayer(n_in=784, n_out=100),\
                   SoftmaxLayer(n_in=100,n_out=10)],\
                   mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    """
    #卷积网络1:单个卷积-混合层
    """
    net = Network([ConvPoolLayer(image_shape=(mini_batch_size,1,28,28),\
                   filter_shape=(20,1,5,5),\
                   poolsize=(2,2)),\
                   FullyConnectedLayer(n_in=20*12*12,n_out=100),\
                   SoftmaxLayer(n_in=100,n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    """
    #卷积网络2:两个卷积-混合层
    """
    net = Network([\
        ConvPoolLayer(image_shape=(mini_batch_size,1,28,28),\
                   filter_shape=(20,1,5,5),\
                   poolsize=(2,2)),\
        ConvPoolLayer(image_shape=(mini_batch_size,20,12,12),\
                   filter_shape=(40,20,5,5),\
                   poolsize=(2,2)),\
        FullyConnectedLayer(n_in=40*4*4,n_out=100),\
        SoftmaxLayer(n_in=100,n_out=10)], mini_batch_size)
    net.SGD(training_data, 60, mini_batch_size, 0.1, validation_data, test_data)
    """
    #卷积网络3(1):两个卷积-混合层,激活函数换为修正线性单元ReLU
    #"""
    net = Network([\
        ConvPoolLayer(image_shape=(mini_batch_size,1,28,28),\
                   filter_shape=(20,1,5,5),\
                   poolsize=(2,2),\
                   activation_fn=ReLU),\
        ConvPoolLayer(image_shape=(mini_batch_size,20,12,12),\
                   filter_shape=(40,20,5,5),\
                   poolsize=(2,2),\
                   activation_fn=ReLU),\
        FullyConnectedLayer(n_in=40*4*4,n_out=100,activation_fn=ReLU),\
        SoftmaxLayer(n_in=100,n_out=10)], mini_batch_size)
    #net.SGD(training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
    net.SGD(training_data, 1, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
    #"""
    #卷积网络3(2):两个卷积-混合层,激活函数换为修正线性单元ReLU + 扩展训练集
    """
    expanded_training_data, validation_data, test_data = network3.load_data_shared("../data/mnist_expanded.pkl.gz")
    net = Network([\
        ConvPoolLayer(image_shape=(mini_batch_size,1,28,28),\
                   filter_shape=(20,1,5,5),\
                   poolsize=(2,2),\
                   activation_fn=ReLU),\
        ConvPoolLayer(image_shape=(mini_batch_size,20,12,12),\
                   filter_shape=(40,20,5,5),\
                   poolsize=(2,2),\
                   activation_fn=ReLU),\
        FullyConnectedLayer(n_in=40*4*4,n_out=100,activation_fn=ReLU),\
        SoftmaxLayer(n_in=100,n_out=10)], mini_batch_size)
    net.SGD(expanded_training_data, 60, mini_batch_size, 0.03, validation_data, test_data, lmbda=0.1)
    """
