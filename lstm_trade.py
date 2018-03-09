#coding=utf-8
'''
基于LSTM机器学习的交易策略
'''
import  pandas as pd
import  numpy as np
import  matplotlib.pyplot as plt
import  tensorflow as tf
import  gmTools
import  talib as ta
import  tushare as ts

MIN_COUNT=500        #开始训练的起始数据量
g_rnn_layers=20       #隐层数量
g_train_times=15      #迭代次数
g_input_columns=7     #最后一列是标签值
g_label_column =g_input_columns
g_output_columns=1
g_learningrate=0.0008         #学习率

#——————————————————导入数据——————————————————————
#return data :open,close,low,high,volume,money,change,label
def read_data_from_file(filepath='dataset_2.csv'):
    f=open(filepath)
    #index_code,date,open,close,low,high,volume,money,change,label
    df=pd.read_csv(f)     #读入数据
    #取第3-10列  open,close,low,high,volume,money,change,label
    return df.iloc[:,2:10].values


#从掘金量化获取K线数据
# return data :open,close,low,high,volume,money,change,label
def read_data_from_gm(stockcode='SZSE.000651',
                      week_in_minutes=30,
                      start='2015-01-01 09:30:00',
                      stop='2018-02-15 15:01:00'):
    #'endtime', 'open', 'high', 'low', 'close', 'volume', 'amount'
    #index_code	date open	close	low	high	volume	money	change	label
    df=gmTools.read_kline(stockcode,week_in_minutes*60,start,stop)     #读入数据
    df['change']=df['close'].pct_change()
    df['label'] = df['close'].shift(-1)

    return df.fillna(0)

# 数据不好用，放弃
def read_data_from_tushare(stockcode='SZSE.000651',
                      week_in_minutes=30,
                      start='2015-01-01 09:30:00',
                      stop='2018-02-15 15:01:00'):
    #'endtime', 'open', 'high', 'low', 'close', 'volume', 'amount'
    #index_code	date open	close	low	high	volume	money	change	label
    df=ts.get_hist_data(stockcode,week_in_minutes*60,start,stop)     #读入数据
    df['change']=df['close'].pct_change()
    df['label'] = df['close'].shift(-1)

    return df.fillna(0)

#生成训练集、测试集
#考虑到真实的训练环境，这里把每批次训练样本数（batch_size）、时间步（time_step）、
#训练集的数量（train_begin,train_end）设定为参数，使得训练更加机动。

#获取训练集  步进为1、长度为time_step的训练数据块，batch_size为一组
def get_train_data(data,batch_size=60,
                   time_step=20,train_begin=0,train_end=5800):
    batch_index=[]
    data_train=data[train_begin:train_end].iloc[:,1:].values

    # 数据归一化处理
    mean = np.mean(data_train, axis=0)
    std = np.std(data_train, axis=0)
    normalized_train_data = (data_train - mean) / std  # 标准化

    train_x,train_y=[],[]   #训练集
    #data matrix: batch_size*time_step*7
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)

       #输入的数据 前7列
       # data matrix:time*7
       x=normalized_train_data[i:i+time_step,:7]

       #y数据真值  第8列  label
       y=normalized_train_data[i:i+time_step,7,np.newaxis]

       train_x.append(x.tolist())
       train_y.append(y.tolist())

    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y


#获取测试集  步进为time_step的测试数据块,
# 与训练数据格式不一致、处理方式也不一致，预测周期越长越不准确
# 数据块不完整时必须用0补充完整
def get_test_data(data,time_step=20,test_begin=5800):
    data_test=data[test_begin:].iloc[:,1:].values

    # 数据归一化处理
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化

    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample

    test_x,test_y=[],[]

    #for i in range(size-1):
    for i in range(1, size - 1):
       #x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
       #y=normalized_test_data[i*time_step:(i+1)*time_step,7]
       #new
       '''
       x=normalized_test_data[(i-1)*time_step:(i)*time_step,:7]
       y=normalized_test_data[(i-1)*time_step:(i)*time_step,7]
       '''
       #todo 考虑用前面的若干天的均值代替未来值进行预测
       x=np.zeros((len(normalized_test_data),7))
       y = np.zeros((len(normalized_test_data),1))
       test_x.append(x.tolist())
       #  扩展test_y列表
       test_y.extend(y)

    #todo 尾数处理
    #test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    #  扩展test_y列表
    #test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())

    return mean,std,test_x,test_y


#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([g_input_columns,g_rnn_layers])),
         'out':tf.Variable(tf.random_normal([g_rnn_layers,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[g_rnn_layers,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']

    # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input=tf.reshape(X,[-1,g_input_columns])
    input_rnn=tf.matmul(input,w_in)+b_in

    # 将tensor转成3维，作为lstm cell的输入
    input_rnn=tf.reshape(input_rnn,[-1,time_step,g_rnn_layers])

    cell=tf.nn.rnn_cell.BasicLSTMCell(g_rnn_layers)

    init_state=cell.zero_state(batch_size,dtype=tf.float32)

    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,
                            initial_state=init_state, dtype=tf.float32)
    
    output=tf.reshape(output_rnn,[-1,g_rnn_layers]) 
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out

    return pred,final_states

#————————————————训练模型————————————————————
def train_lstm(stock_code,weeks,data,train_times=g_train_times,batch_size=20,
               time_step=10,train_begin=2000,train_end=5800,stopDT=''):

    X=tf.placeholder(tf.float32, shape=[None,time_step,g_input_columns])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,g_output_columns])

    try:
        batch_index,train_x,train_y=get_train_data(data,
                         batch_size,time_step,train_begin,train_end)

        var_name="sec_lstm_{0}_{1}_{2}".format(stock_code[5:],weeks,stopDT[:10])
        with tf.variable_scope(var_name):
            pred,_=lstm(X)

        #误差函数   计算平均值，似乎不是最佳方案
        loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))

        train_step=tf.train.AdamOptimizer(g_learningrate).minimize(loss)

        saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
        model_path='model\\{0}-{1}\\model.ckpt'.\
            format(stock_code, weeks)

        with tf.Session() as sess:
            try:
                # 定义了存储模型的文件路径，即：当前运行的python文件路径下，文件名为model.ckpt
                saver.restore(sess, model_path)
                #读取继续训练的开始位置  通过控制读入数据进行处理
            except:
                # 如果是第一次运行，通过init告知tf加载并初始化变量
                print("未加载模型参数，文件被删除或者第一次运行")
                sess.run(tf.global_variables_initializer())
                if  len(data) - train_begin < MIN_COUNT:
                    # 数据太少，不做分析
                    return

            #todo 增加利用已计算结果的处理  在主控程序控制好
            for i in range(train_times):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
                for step in range(len(batch_index)-1):
                    _,loss_=sess.run([train_step,loss],
                            feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],
                                       Y:train_y[batch_index[step]:batch_index[step+1]]})

                #print("Number of iterations:",i," loss:",loss_)

            print("model_save: ",saver.save(sess,model_path))
            # 保存结束训练的位置  通过控制读入数据进行处理

            print("The train has finished,loss:",loss_)

    except:
        pass

#————————————————预测模型————————————————————
def prediction(stock_code,weeks,data,time_step=10,pred_begin=5800,stopDT=''):
    pred_step=int(time_step)
    X=tf.placeholder(tf.float32, shape=[None,pred_step,g_input_columns])

    mean,std,test_x,test_y=get_test_data(data=data,
                    time_step=pred_step,test_begin=pred_begin)

    var_name = "sec_lstm_{0}_{1}_{2}".format(stock_code[5:], weeks, stopDT[:10])
    with tf.variable_scope(var_name,reuse=True):
        pred,_=lstm(X)

    saver=tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint\
            ('model\\'+stock_code+'-'+str(weeks))

        saver.restore(sess, module_file)

        test_predict = []
        for step in range(len(test_x)):
          #预测阶段能否持续优化模型？？？
          try:
              prob=sess.run(pred,feed_dict={X:[test_x[step]]})
              predict=prob.reshape((-1))
              test_predict.extend(predict)
          except:
              pass

        # 数据恢复为原来的真值
        test_y=np.array(test_y)*std[g_label_column]+mean[g_label_column]
        test_predict=np.array(test_predict)*std[g_label_column]+mean[g_label_column]

        # 偏差程度
        acc=np.average(np.abs(test_predict
                              -test_y[:len(test_predict)])/test_y[:len(test_predict)])
        print("The accuracy of this predict:",acc)

        try:
            #增加预测周期内的买卖组合分析、判断
            best_buy_point,betst_buy_price=test_predict.argmin(),test_predict.min()
            best_sell_point,betst_sell_price=test_predict.argmax(),test_predict.max()

            #判断是否为多头趋势
            if best_sell_point-best_buy_point  < 0:
                best_buy_point = test_predict[:best_sell_point].argmin()
                betst_buy_price = test_predict[:best_sell_point].min()

            # 以折线图表示结果 figsize=(20, 15)
            plt.figure()
            plot_predict = plt.plot(list(range(len(test_predict))),
                                    test_predict, color='b', label='predict')
            plot_test_y = plt.plot(list(range(len(test_y))),
                                   test_y, color='r', label='true')

            if best_sell_point-best_buy_point>=240/weeks:
                #最少要持仓一天
                legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
                islong = True
                real_sell_price = test_y[best_sell_point]
                real_buy_price = test_y[best_buy_point]
                title = 'buy {0}--{1:0>3}--gain({2:.2f})({3:.2f})\n' \
                        'buy({4:.2f})({5:.2f}),sell({6:.2f})({7:.2f})' \
                    .format(stock_code[5:], weeks,
                            100 * betst_sell_price / betst_buy_price - 100,
                            100 * real_sell_price / real_buy_price - 100,
                            betst_buy_price, real_buy_price,
                            betst_sell_price, real_sell_price)
            else :
                islong = False
                legend = plt.legend(loc='lower left', shadow=True, fontsize='x-large')
                best_buy_point= test_predict.argmin()
                betst_buy_price=test_predict.min()
                real_sell_price = test_y[best_sell_point]
                real_buy_price = test_y[best_buy_point]

                if best_buy_point-best_sell_point  >= 240 / weeks:
                    title = 'sell {0}--{1:0>3}--gain({2:.2f})({3:.2f})\n' \
                            'sell({4:.2f})({5:.2f}),buy({6:.2f})({7:.2f})' \
                        .format(stock_code[5:], weeks,
                                100 * betst_sell_price / betst_buy_price - 100,
                                100 * real_sell_price / real_buy_price - 100,
                                betst_sell_price, real_sell_price,
                                betst_buy_price, real_buy_price)
                else:
                    title = '[{0}--{1:0>3}] no use of buy or sellsell ' \
                        .format(stock_code[5:], weeks)


            title='date[{0}--{1}] {2}'.format(startDT[:10],stopDT[:10] ,title)
            print (title)


            plt.title(title)

            plt.annotate('buy', xy=(best_buy_point, betst_buy_price),
                         xytext=(best_buy_point*1.1, betst_buy_price),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         )

            plt.annotate('sell', xy=(best_sell_point, betst_sell_price),
                         xytext=(best_sell_point * 0.9, betst_sell_price ),
                         arrowprops=dict(facecolor='green', shrink=0.05),
                         )

            # Put a nicer background color on the legend.  ????
            if islong==False:
                legend.get_frame().set_facecolor('green')

            file = 'fig\\{0}--{1:0>3}--{2}.png'.format(stock_code, weeks,stopDT[:10])
            plt.savefig(file)
            plt.close()
            # plt.show()
        except:
            pass



def predict_buy_sell(stock_code,weeks,startDT,stopDT):
    batch_size = 20
    time_step = 10
    analyse_block = batch_size * time_step

    g_data=None
    g_data=read_data_from_gm(stock_code,weeks,startDT,stopDT)  #read_data_from_file()
    #为确保数据的完整性，需规划好参与训练与测试的数据集

    data_size=len(g_data)

    #确保数据块的完整性，且最新的数据能用于训练与预测
    remainder=data_size % analyse_block

    '''
    if remainder>0:
        #用最新数据扩充
        tmp=pd.DataFrame()
        g_data=g_data[remainder:]
        data_size = len(g_data)
    '''

    data_blocks=int(data_size/analyse_block)
    # stop on 90% ,prediction on the lastest 10%
    train_end =(analyse_block-2) * data_blocks

    train_begin=int(analyse_block*data_blocks*0.2)   #from 10% on

    if train_end<0 :
        #数据太少，不做分析
        return

    train_lstm(stock_code,weeks,g_data,
        g_train_times,batch_size,time_step,train_begin,train_end,stopDT)

    prediction(stock_code,weeks,g_data,time_step,train_end,stopDT)

date_list=[
           ['2015-01-01 09:30:00','2017-01-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-02-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-03-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-04-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-05-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-06-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-07-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-08-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-10-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-11-15 15:01:00'],
           ['2015-01-01 09:30:00','2017-12-15 15:01:00'],
           ['2015-01-01 09:30:00','2018-01-15 15:01:00'],
           ['2015-01-01 09:30:00','2018-02-15 15:01:00']
        ]
#'SZSE.002465','SZSE.000651','SZSE.002460','SZSE.000661','SZSE.000725','SZSE.000001',
#            'SHSE.601318','SHSE.600196','SHSE.600519',
'''
000001,000063,000333	
000513,000538	
000651,000661	
000710,000725	
000963,002008	
002019,002152	
002415,002460	
002465，002594	
300003，300072	
600000，600016	
600018，600085	
600196，600271	
600276，600309	
600332，600380	
600436，600518	
600519，600584	
600585，600690	
600703，600887	
600900，601009	
601169，601288	
601318，601607	
601766，603288	
'''
stock_codes=[#'SZSE.002465','SZSE.000651','SZSE.002460',
            'SHSE.600297','SZSE.002241','SHSE.601318','SHSE.600196','SHSE.600519','SZSE.002415',
            'SHSE.000063','SHSE.000333','SHSE.000513','SHSE.000538'
             'SZSE.000661','SZSE.000725','SZSE.000001','SHSE.600887','SHSE.601009']
weeks=[15,30,60,120,5,2]
g_startDate='2015-01-01'
g_stopDate='2018-05-08'
g_startTime=' 09:30:00'
g_stopTime=' 15:01:00'

for stock in stock_codes:
    for week in weeks:
            startDT=g_startDate+g_startTime
            for analyse_date in date_list:
                try:
                    stopDT=analyse_date[1]
                    print('stock:',stock,'week:',week,',start:',startDT[:10],',stop:',stopDT[:10])
                    predict_buy_sell(stock, week,startDT,stopDT)
                    #startDT = stopDT
                except:
                    print('error stock:', stock, 'week:', week, ',date:', analyse_date)
                    #startDT = stopDT
                    continue