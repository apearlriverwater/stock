# coding=utf-8

import tensorflow as tf
import numpy  as np
import gmTools_v2 as gmTools
import time
import  matplotlib.pyplot as plt
import pandas as pd
import datetime
import struct

""" 
2018-02-28:
    1)考虑把目前的基于股票的串行训练方式改为全标的按时间段统一训练的平行模式，防止利用了未来信息；
    由于各标的情况不一，同一时段内部分标的可能存在停牌情况，数据项不一定相等，采用在特定时段内按
    标的逐一进行训练的方法，时间分为：训练时间段【2015-01-01 至2016-06-01】，
    回测时间段【2016-06-02 至2017-12-31，暂不考虑中间要空g_max_holding_weeks项数据】，
    实盘时间段【2018年起】；
    2）考虑增加图形显示当前分析股票的价格走势图；
    
2018-02-27:
    1)PyCharm集成开发环境非常耗内存，程序运行时切换到IDLE（Ipython)运行，内存占用显著下降。
    2)由于大数据量运算时占用内存很大，考虑对处理过的数据进行释放，减少内存占用。
    3)增加大盘指数（SHSE.000001 上证综指，SHSE.000002 上证A指，SZSE.399001 深证成指，
      SZSE.399005 中小板指，SZSE.399006 创业板指）、全市场动态数据，评估分析选择沪深300（SZSE.399300）、
     上证50（SHSE.000016）。
    4)同一模型不能在同一次运行中多次加载。 
    
2018-02-25:
    1）在基本功能的基础上，增加基于特定stock的模型管理，图中增加用于记录本模型适应的stock代码、
    训练过的时间范围、频率、模型平均准确率等信息，后续使用时先判断是否有可用的模型，如果有即加载
    模型后继续训练或直接使用；
    2）模型应能持续优化，使用时间越长、精度应该越高；
    3)无法解决utc时间到字符串的转后的结果在gm读取信息中的使用问题，int(week*60)强制类型转换
    4)以股票为单位进行循环训练，数据项超过6000即停止K线数据读取，处理完毕后继续进行；
    
2018-02-22：
    基于本框架进行交易模型设计。首先立足特定股票的波段操作，基本思路：
    1）建立趋势模型：当前时点t的n周期走势划分为11级，对应0，+-1，+-2，+-3，+-4，+-5，
    5表示大于等于5%涨幅，-5小于等于5%跌幅；可扩充为k的倍数，但级数保持11级；【0225已实现】
    1 week的交易数据数组类似一张图片的数据数组；
    2）基本模型：利用ochl，vol，amount；【0225已实现】
      增强模型：上证、深证、沪深300主要大盘指数进行学习与预测,基于level2的
      买卖资金量，talib支持的其他技术指标；
    3）基本模型基于5分钟数据进行学习，模拟盘中利用1分钟数据不断增强模型的适应性；
      数据矩阵[,9] ;【0225已实现】
    4）买卖时机：全市场股票升跌数据统计，用于指导买卖，特例：每次市场大跌是否有征兆？
      
----------------------------------------------------------------------------------
非常清晰明了的介绍，适合学习模仿。
首先载入Tensorflow，并设置训练的最大步数为1000,学习率为0.001,dropout的保留比率为0.9。 
同时，设置MNIST数据下载地址data_dir和汇总数据的日志存放路径log_dir。 
这里的日志路径log_dir非常重要，会存放所有汇总数据供Tensorflow展示。 
"""

#沪深300（SZSE.399300）、
#     上证50（SHSE.000016）
STOCK_BLOCK='SHSE.000016'

MAX_HOLDING=5
BUY_GATE=7
SELL_GATE=3
BUY_FEE=1E-4
SELL_FEE=1E-4
DAY_SECONDS=24*60*60
MAX_STOCKS=5
g_train_startDT='2005-01-01'  # oldest start 2015-01-01
g_train_stopDT='2018-01-01'
g_backtest_stopDT='2019-01-01'

g_max_step = 20000
g_learning_rate = 0.001
g_dropout = 0.9

#策略参数
g_week=240  #freqency
g_max_holding_days=15

g_input_columns=6
g_trade_minutes=240
g_week_in_trade_day=int(g_trade_minutes/g_week)
g_look_back_weeks=max(10,g_week_in_trade_day*2)  #回溯分析的周期数
g_max_holding_weeks=g_week_in_trade_day*g_max_holding_days  #用于生成持仓周期内的收益等级


g_max_stage=11  #持仓周期内收益等级
g_stage_rate=2  if g_week>30 else 1#持仓周期内收益等级差

g_log_dir = 'logs/week{0}hold{1}days'.format(g_week,g_max_holding_days)


g_current_train_stop=0                  #当前测试数据结束位置
g_test_stop=0                   #当前实时数据结束位置
g_stock_current_price_list=0

train_x=0
train_y=0
g_trade_startTime=' 09:30:00'
g_trade_stopTime=' 15:00:10'

g_test_securities=["SZSE.002415","SZSE.000333","SZSE.002460",
                   "SZSE.000001","SZSE.002465","SZSE.002466",
"SZSE.000651","SZSE.000725","SZSE.002152","SZSE.000538","SZSE.300072",
"SHSE.603288","SHSE.600703","SHSE.600271", "SHSE.600690", "SHSE.600585", "SHSE.600271",
"SHSE.600000","SHSE.600519"]

#标签数据生成，自动转成行向量 0-10共11级，级差1%
def make_stage(x):
    x=int(100*x/g_stage_rate)
    if abs(x)<5:
        x = x+5
    elif x>4:
        x = 10
    else:
        x = 0

    tmp = np.zeros(g_max_stage, dtype=np.int)
    tmp[x]=1
    return tmp

# 获取测试集  步进为time_step的测试数据块,
# 与训练数据格式不一致、处理方式也不一致，预测周期越长越不准确
# 数据块不完整时必须用0补充完整
def get_test_data(data, normalized_data,
    look_back_weeks=g_look_back_weeks):
    train_x, train_y = [], []
    start=look_back_weeks
    #for i in range(look_back_weeks, len(data)):
    for i in range(look_back_weeks,int(len(data))):
        x = normalized_data.iloc[start- look_back_weeks:start, :]
        y = data.iloc[start-look_back_weeks:start, -1]
        start+=1 #look_back_weeks

        train_x.append(x.values.tolist())

        #test_y.extend(y.values.tolist())
        train_y.append(y.values.tolist())

    return train_x, train_y

def create_market_data(stock,start_DateTime,stop_DateTime,
        week=g_week,look_back_weeks=g_look_back_weeks):

    global  g_market_train_data,g_input_columns,\
        g_normalized_data,g_max_step,train_x,train_y

    g_market_train_data=gmTools.read_kline(stock,int(week*60),
            start_DateTime,stop_DateTime,50000)    #训练数据
    if len(g_market_train_data)==0:
        return
    #预测look_back_weeks周期后的收益
    g_market_train_data['label']=g_market_train_data['close'].pct_change(look_back_weeks)
    g_market_train_data['label']=g_market_train_data['label'].shift(-look_back_weeks)
    #将数据总项数整理成g_max_holding_weeks的整数倍
    #tmp = len(g_market_train_data)%g_max_holding_weeks+g_max_holding_weeks
    #g_market_train_data =g_market_train_data[tmp:]
    g_market_train_data['label'] =g_market_train_data['label'].fillna(0)
    g_market_train_data['label'] = g_market_train_data['label'].apply(make_stage)
    data_tmp = g_market_train_data.iloc[:, 1:-1]
    #todo  加入其他的技术分析指标

    # 数据归一化处理

    mean = np.mean(data_tmp, axis=0)
    std = np.std(data_tmp, axis=0)
    g_normalized_data = (data_tmp - mean) / std  # 标准化

    g_input_columns=len(data_tmp.columns)

    cols=['eob', 'close','label']
    g_market_train_data = g_market_train_data[cols]
    g_max_step = len(g_market_train_data)

    #数据规整为look_back_weeks的整数倍
    remainder=len(g_market_train_data)%look_back_weeks
    g_market_train_data=g_market_train_data[remainder:]
    g_normalized_data = g_normalized_data[remainder:]

    train_x,train_y=get_test_data(g_market_train_data,
            g_normalized_data,look_back_weeks)


def create_market_last_n_data(stocks, count, stop_DateTime,
                       week=g_week, look_back_weeks=g_look_back_weeks):
    global g_stock_current_price_list, g_input_columns, \
        g_normalized_data, g_max_step, train_x, train_y

    market_train_data = gmTools.read_last_n_kline(stocks, int(week * 60),
        count, stop_DateTime)  # 训练数据

    g_max_step = len(market_train_data)

    if g_max_step == 0:
        return
    g_stock_current_price_list=[]
    train_x = []
    train_y = []

    #以排序后的股票代码为序保存g_market_train_data['label'] = g_market_train_data['label'].apply(make_stage)
    for kline in market_train_data:
        stock,kdata=kline['code'],kline['kdata']

        data_tmp = kdata.iloc[:, 1:]
        # todo  加入其他的技术分析指标

        # 数据归一化处理
        mean = np.mean(data_tmp, axis=0)
        std = np.std(data_tmp, axis=0)
        g_normalized_data = (data_tmp - mean) / std  # 标准化

        g_input_columns = len(data_tmp.columns)

        cols = ['eob', 'close']
        g_stock_current_price_list.append({'code':stock,'time_close':kdata[cols][-1:].values.tolist()})

        y=int(kdata['close'][len(kdata)-1]*100/kdata['close'][0]-100)
        train_x.append(g_normalized_data.values.tolist())
        #shape(?,g_max_tage)
        train_y.append([make_stage(y)])


# 定义对Variable变量的数据汇总函数
""" 
计算出var的mean,stddev,max和min， 
对这些标量数据使用tf.summary.scalar进行记录和汇总。 
同时，使用tf.summary.histogram直接记录变量var的直方图。 
"""
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

# 定义神经网络模型参数的初始化方法，
# 权重依然使用常用的truncated_normal进行初始化，偏置则赋值为0.1
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 设计一个MLP多层神经网络来训练数据，在每一层中都会对模型参数进行数据汇总。
""" 
定一个创建一层神经网络并进行数据汇总的函数nn_layer。 
这个函数的输入参数有输入数据input_tensor,输入的维度input_dim,
输出的维度output_dim和层名称layer_name，激活函数act则默认使用Relu。 
在函数内，显示初始化这层神经网络的权重和偏置，并使用前面定义的
variable_summaries对variable进行数据汇总。 
然后对输入做矩阵乘法并加上偏置，再将未进行激活的结果使用tf.summary.histogram统计直方图。 
同时，在使用激活函数后，再使用tf.summary.histogram统计一次。 
"""
def nn_layer(input_tensor, input_dim,
             output_dim, layer_name,act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable([input_dim, output_dim])
            #variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            #variable_summaries(biases)

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)

        activations = act(preactivate, name='actvations')
        tf.summary.histogram('activations', activations)
        return activations

""" 
使用刚定义好的nn_layer创建一层神经网络，输入维度（1*g_input_columns），
输出的维度是隐藏节点数500. 
再创建一个Droput层，并使用tf.summary.scalar记录keep_prob。
然后再使用nn_layer定义神经网络的输出层，激活函数为全等映射，
此层暂时不使用softmax,在后面会处理。 
"""

g_nn_hidden_nodes=0
hidden1=0
x=0
y=0
y1=0
cross_entropy=0
train_step=0
buy=0
sell=0
accuracy=0
merged=0
train_writer=0
test_writer=0
keep_prob=0
sell_prediction=0
buy_prediction=0
valid_accuracy=0
valid_accuracy2=0
model_code=0
model_last_utc=0
model_next_train_utc=0
model_week=0
reward_prediction=0

def setup_tensor(sess,stock,week,last_utc,next_train_time=0):
    global  g_nn_hidden_nodes,hidden1,cross_entropy,\
        train_step,buy,sell,accuracy,merged,train_writer,\
        test_writer,keep_prob,x,y,y1,sell_prediction,\
        buy_prediction,valid_accuracy,valid_accuracy2,\
        model_code,model_last_utc,model_week,\
        model_next_train_utc,reward_prediction

    """ 
    为了在TensorBoard中展示节点名称，设计网络时会常使用tf.name_scope限制命名空间， 
    在这个with下所有的节点都会自动命名为input/xxx这样的格式。 
    定义输入x和y的placeholder，并将输入的一维数据变形为28×28的图片存储到另一个tensor， 
    这样就可以使用tf.summary.image将图片数据汇总给TensorBoard展示了。 
    """

    #定义网络参数
    with tf.name_scope("model_vars"):
        model_code=tf.Variable(stock,dtype=tf.string,trainable=False,name="model_code")
        model_week = tf.Variable(week, dtype=tf.int32, trainable=False, name="model_week")
        model_last_utc=tf.Variable(last_utc, dtype=tf.int64,
                    trainable=False, name="model_last_utc")
        model_next_train_utc = tf.Variable(next_train_time, dtype=tf.int64,
                                     trainable=False, name="model_next_train_utc")

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,
            1 * g_input_columns], name='x_input')
        y = tf.placeholder(tf.float32, [None, g_max_stage], name='y_input')

    g_nn_hidden_nodes=800
    hidden1 = nn_layer(x, 1*g_input_columns, g_nn_hidden_nodes, 'layer1')

    with tf.name_scope('g_dropout'):
        keep_prob = tf.placeholder(tf.float32)
        #tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y1 = nn_layer(dropped, g_nn_hidden_nodes, g_max_stage, 'layer2', act=tf.identity)

    """ 
    这里使用tf.nn.softmax_cross_entropy_with_logits()
    对前面输出层的结果进行softmax处理并计算交叉熵损失cross_entropy。 
    计算平均损失，并使用tf.summary.saclar进行统计汇总。 
    """
    with tf.name_scope('prediction'):
        # 绝对精度  完全相等的预测结果
        correct_prediction = tf.equal(tf.argmax(y1, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        #ones= tf.constant(1.0, shape=[g_max_stage])
        #loss=tf.reduce_sum(1-accuracy)

        # 误差在1个单位增益范围内的结果
        valid_prediction = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), g_stage_rate)
        error=tf.reduce_mean(tf.cast(tf.abs( tf.argmax(y1, 1)-tf.argmax(y, 1)),tf.float32))
        valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction, tf.float32))
        tf.summary.scalar('valid_accuracy', valid_accuracy)
        tf.summary.scalar('error', error)

        # 误差在两个单位增益范围内的结果
        valid_prediction2 = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 2 * g_stage_rate)
        valid_accuracy2 = tf.reduce_mean(tf.cast(valid_prediction2, tf.float32))
        tf.summary.scalar('valid_accuracy2', valid_accuracy2)


        diff = tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y)
        #diff = tf.nn.softmax_cross_entropy_with_logits(logits=prediction_mean, labels=true_mean)
        cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

    """ 
    使用Adma优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuray， 
    再使用tf.summary.scalar对accuracy进行统计汇总。 
    train_step = tf.train.AdamOptimizer(g_learning_rate).minimize(cross_entropy)
    """
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(g_learning_rate).minimize(cross_entropy)

    """ 
    由于之前定义了非常多的tf.summary的汇总操作，一一执行这些操作态麻烦， 
    所以这里使用tf.summary.merger_all()直接获取所有汇总操作，以便后面执行。 
    然后，定义两个tf.summary.FileWrite(文件记录器)在不同的子目录，
    分别用来存放训练和测试的日志数据。 
    同时，将Session的计算图sess.graph加入训练过程的记录器，
    这样在TensorBoard的GRAPHS窗口中就能展示整个计算图的可视化效果。 
    最后使用tf.global_variables_initializer().run()初始化全部变量。 
    """

    merged = tf.summary.merge_all()
    #自动生成工程需要的文件目录
    train_writer = tf.summary.FileWriter(g_log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(g_log_dir + '/test')
    test_figure = tf.summary.FileWriter(g_log_dir + '/fig')
    tf.global_variables_initializer().run()


def feed_dict(train):
    global  g_current_train_stop

    xs = train_x[g_current_train_stop]
    ys = train_y[g_current_train_stop]

    #todo  数据取完后如何处理？  直接退出运行,需支持对未来收益的预测
    if train:
        k = g_dropout
        g_current_train_stop += 1
    else:
        k = 1.0

    return {x: xs, y: ys, keep_prob: k}


# 实际执行具体的训练，测试及日志记录的操作
""" 
首先，使用tf.train.Saver()创建模型的保存器。 
然后，进入训练的循环中，每隔10步执行一次merged（数据汇总），
accuracy（求测试集上的预测准确率）操作， 
并使应test_write.add_summary将汇总结果summary和循环步数i写入日志文件; 
同时每隔100步，使用tf.RunOption定义Tensorflow运行选项，其中设置trace_level为FULL——TRACE, 
并使用tf.RunMetadata()定义Tensorflow运行的元信息， 
这样可以记录训练是运算时间和内存占用等方面的信息. 
再执行merged数据汇总操作和train_step训练操作，将汇总summary和训练元信息run_metadata添加到train_writer. 
平时，则执行merged操作和train_step操作，并添加summary到trian_writer。 
所有训练全部结束后，关闭train_writer和test_writer。 
"""
def train_model( week=g_week,look_back_weeks=g_look_back_weeks):
    global  g_train_startDT,g_current_train_stop,g_market_train_data,\
        g_normalized_data,g_max_step,train_x,train_y

    model_path = g_log_dir

    sess = tf.InteractiveSession()
    setup_tensor(sess, STOCK_BLOCK, g_week,0)
    saver = tf.train.Saver()

    model_file = tf.train.latest_checkpoint(model_path)

    model_trained = False
    if model_file:
        try:
            saver.restore(sess, model_file)
            week, code, last_utc = sess.run([model_week, model_code, model_last_utc])
            # code string ,返回是bytes类型，需要转换
            print("restore from model code=%s,week=%d,last datetime %s" % (
                code, week, gmTools.timestamp_datetime(last_utc)))
        except:
            pass

    ii=0
    total_count=0

    for stock in g_test_securities:
        ii += 1
        print("\n[%s]start training model [%s] %.2f%%"%(stock,
                time.strftime('%Y-%m-%d %H:%M:%S',
                time.localtime(time.time())),
                ii*100/len(g_test_securities)))


        startDT =g_train_startDT+g_trade_startTime

        create_market_data(stock=stock,
                           start_DateTime=startDT,
                           stop_DateTime=g_train_stopDT + g_trade_stopTime,
                           week=week, look_back_weeks=g_look_back_weeks)

        g_max_step = len(g_market_train_data)
        print('log dir:%s , total items :%d' % (g_log_dir, g_max_step))

        if g_max_step<=g_week_in_trade_day:
            continue

        train_count=int(g_max_step) #/look_back_weeks)

        g_current_train_stop = 0
        i=g_current_train_stop

        print("training %s" % (g_market_train_data.iloc[0,0].strftime('%Y-%m-%d %H:%M:%S')))

        train_writer = tf.summary.FileWriter(g_log_dir + '/train/'+stock, sess.graph)

        for i in range( look_back_weeks,train_count - 1):
            feed_dict_data=feed_dict(True)
            #
            summary,_= sess.run([merged,train_step],feed_dict=feed_dict_data)

            if total_count% 20==0:
                train_writer.add_summary(summary, i)
                #tmp=sess.run(cross_entropy,feed_dict=feed_dict_data)

            total_count += 1


        g_market_train_data = 0
        g_normalized_data = 0
        g_max_step = 0
        train_x = 0
        train_y = 0

    saver.save(sess, model_path + "/model.ckpt")  # save模型
    sess.close()
    print('total train %d steps'%(total_count))

'''
    在价格走势图显示买卖点信息
'''
def draw_bs_on_kline(stock,kdata,buy_utc,sell_utc,week=g_week):
    # 以折线图表示结果 figsize=(20, 15)
    try:
        plt.figure()
        data=kdata['close'].values.tolist()

        plot = plt.plot(list(range(len(kdata))),
                    data, color='b', label='close')
        utclist=kdata['eob'].values.tolist()

        buy_time=buy_utc.strftime('%Y-%m-%d %H:%M:%S')
        sell_time=sell_utc.strftime('%Y-%m-%d %H:%M:%S')

        title = ' {2} week={3} \n [{0}--{1}]'.format(buy_time, sell_time, stock,week)

        plt.title(title)

        x=utclist.index(buy_utc)
        y=data[x]

        plt.annotate('buy', xy=(x, y),
                     xytext=(x * 1.1, y),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     )
        x = utclist.index(sell_utc)
        y = data[x]

        plt.annotate('sell', xy=(x, y),
                     xytext=(x * 0.9, y),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     )
    except:
        pass

    buy_time = buy_utc.strftime('%Y-%m-%d %H-%M-%S')
    sell_time = sell_utc.strftime('%Y-%m-%d %H-%M-%S')

    file = '{3}/{0}--{1}--{2}.png'.format(stock, buy_time, sell_time,g_log_dir + '/fig' )
    plt.savefig(file)
    plt.close()

'''
    由于掘金返回批量股票K线数据很慢，采用集中获取买卖点信息后统一处理，
    自行计算买卖数据，自行统计盈利情况
'''
def backtest_model( week=g_week,look_back_weeks=g_look_back_weeks,
                    startDT =g_train_stopDT,stopDT=g_backtest_stopDT):
    global  g_train_startDT,g_current_train_stop,g_market_train_data,\
        g_normalized_data,g_max_step,train_x,train_y

    print("\nstart backtest_model [%s]" % (
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    code_4_buy=[]
    code_4_sell=[]
    ii=0

    for stock in g_test_securities:
        print("\n[%s]start backtesting [%s] %.2f%%"%(stock,
                time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),ii*100/len(g_test_securities)))
        ii+=1

        create_market_data(stock=stock,
                           start_DateTime=startDT+g_trade_startTime,
                           stop_DateTime=stopDT + g_trade_stopTime,
                           week=week, look_back_weeks=g_look_back_weeks)

        g_max_step = len(g_market_train_data)
        if g_max_step<look_back_weeks:
            continue

        train_count=int(g_max_step/look_back_weeks)
        print('log dir:%s , total items :%d'%(g_log_dir,g_max_step))

        g_current_train_stop = 0
        i=g_current_train_stop
        try:
            print("backtesting %s" % (g_market_train_data.iloc[0,0].strftime('%Y-%m-%d %H:%M:%S')))
        except:
            print('error backtesting %s ' %(stock))
            pass

        last_reward=[5,5]
        buy_on = False  # 连续出现的buy信号不处理
        buy_point=0

        #detect buy-sell signal
        for i in range(train_count-1):
            feed_dict_data = feed_dict(False)
            reward = sess.run(reward_prediction, feed_dict=feed_dict_data)

            # todo 简单判断未来走势未必合理，是否考虑看均线趋势？
            index=look_back_weeks*i
            last_reward=last_reward+reward.tolist()

            for j in range(2,look_back_weeks+2):
                if not buy_on and  last_reward[j-1] >= BUY_GATE and last_reward[j-2] >= BUY_GATE:
                    buy_on=True
                    buy_point=index+j
                    code_4_buy.append({'code': stock,'time':g_market_train_data.iloc[buy_point,0],
                            'price':g_market_train_data.iloc[buy_point,1],'reward':last_reward[j-1]})

                if buy_on and index+j-buy_point>g_week_in_trade_day:
                    if(last_reward[j-1] <= SELL_GATE and last_reward[j-2] <= SELL_GATE) \
                        or index+j-buy_point>g_max_holding_weeks: #arrive max holding weeks
                        buy_on=False
                        code_4_sell.append({'code': stock,'time':g_market_train_data.iloc[index+j,0],
                                            'price':g_market_train_data.iloc[index+j,1],'reward':last_reward[j-1]})

            last_reward=last_reward[-2:]

            # train the model
            feed_dict_data = feed_dict(True)
            summary,_,= sess.run([merged,train_step],feed_dict=feed_dict_data)
            #train_writer.add_summary(summary, i)


        g_market_train_data = 0
        g_normalized_data = 0
        g_max_step = 0
        train_x = 0
        train_y = 0

    #对买卖列表按时间顺序进行排序，按时间段进行买卖点分析
    code_4_buy.sort(key=lambda i: (i['time'],i['reward']))
    code_4_sell.sort(key=lambda i: (i['time'],i['reward']))

    #process BS point
    buy_index=0
    sell_index=0
    holding=[]
    holdings=[]
    amount=1e6
    is_sell=True

    bs_index_changed = False

    #process buy sell point
    while len(code_4_sell[sell_index:])>0 or len(code_4_buy[buy_index:])>0:
        utc_time = int(time.time())
        #from the min time on
        try:
            if buy_index<len(code_4_buy):
                utc_time=code_4_buy[buy_index]['time']
                is_sell=False

            if sell_index<len(code_4_sell):
                utc_time=min(code_4_sell[sell_index]['time'],utc_time)
                is_sell = True

            buy_list = []
            for item in code_4_buy[buy_index:]:
                if item['time'] == utc_time:
                    buy_list.append(item)
                    buy_index += 1
                    bs_index_changed=True
                else:
                    break

            if len(buy_list) > 0 and len(holding) < MAX_HOLDING:
                buy_list.sort(key=lambda i: i['reward'], reverse=True)
                buy_list = buy_list[:MAX_HOLDING]

                for item in buy_list:
                    if len(holding) >= MAX_HOLDING:
                        break
                    elif not item['code'] in holdings :  # buy only once
                        money = amount / (MAX_HOLDING - len(holding)) / 1.1
                        vol = int(money / (item['price'] * (1 + BUY_FEE) * 100)) * 100
                        if vol > 0:
                            tmp=vol * item['price'] * (1 + BUY_FEE)
                            amount -= tmp
                            holdings.append(item['code'])

                            start_datetime=time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(utc_time-DAY_SECONDS*g_max_holding_days))
                            must_sell_datetime = time.strftime('%Y-%m-%d %H:%M:%S',
                                        time.localtime(utc_time+DAY_SECONDS *( g_max_holding_days*4)))
                            # TODO ADD K DATA TO HONGDING AND DETECT LOSS
                            k_data = gmTools.read_kline(item['code'], int(week * 60),
                                         start_datetime, must_sell_datetime)  # k line 数据

                            cols = ['eob', 'close']
                            k_data = k_data[cols]

                            holding.append({'code': item['code'], 'price': item['price'],
                                            'vol': vol, 'time': utc_time,'kdata':k_data})
                            print("[%s] buy %s vol=%d,price=%.2f,amt=%.2f,nav=%.2f" %
                                  (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(utc_time)),
                                   item['code'], vol, item['price'],tmp,amount))

            if len(holding) > 0:
                sell_list = []
                for item in code_4_sell[sell_index:]:
                    if item['time'] == utc_time and item['code'] in holdings:
                        i = holdings.index(item['code'])
                        # sell
                        hold_time = utc_time - holding[i]['time']
                        if hold_time > DAY_SECONDS:
                            # stop loss or had hold for g_max_holding_days
                            if item['price'] / holding[i]['price'] < 0.9 \
                                    or hold_time > DAY_SECONDS * g_max_holding_days:
                                vol=holding[i]['vol']
                                tmp=vol * item['price'] * (1 - SELL_FEE)
                                amount += tmp
                                # todo save bs point in the graph
                                draw_bs_on_kline(holding[i]['code'],
                                        holding[i]['kdata'],holding[i]['time'],utc_time)

                                print("[%s] sell %s vol=%d,bs price=[%.2f--%.2f],amt=%.2f,reward=%d,nav=%.2f" %
                                      (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(utc_time)),
                                       item['code'], vol, holding[i]['price'],item['price'],tmp,
                                       int(item['price']*100/holding[i]['price']-100),amount))

                                holding.pop(i)
                                holdings.pop(i)

                        sell_index += 1
                        bs_index_changed = True
                    else:
                        continue

            if bs_index_changed==False:
                if is_sell:
                    sell_index+=1
                else:
                    buy_index+=1
            else:
                bs_index_changed = False

        except:
            print('error backtest_model')
            pass

    saver.save(sess, model_path + "/model.ckpt")  # save模型
    sess.close()

    print("\nstop backtest_model [%s]" % (
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

#返回字典列表
def get_bs_list( stop_dt='',week=g_week,
    look_back_weeks=g_look_back_weeks,count=g_look_back_weeks):
    global g_train_startDT, g_current_train_stop, g_market_train_data, \
        g_normalized_data, g_max_step, train_x, train_y

    now = '[{0}] create_market_last_n_data'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(now)

    create_market_last_n_data(g_test_securities,max(count,look_back_weeks),
        stop_dt,week,look_back_weeks)

    code_4_buy=[]
    code_4_sell=[]
    g_max_step = 0
    g_current_train_stop = 0

    for item in g_stock_current_price_list:
        stock=item['code']
        #print("\n[%s]start analysing [%s]" % (stock,
        #    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

        feed_dict_data = feed_dict(False)
        reward = sess.run( reward_prediction, feed_dict=feed_dict_data)

        #todo 简单判断未来走势未必合理，是否考虑看均线趋势？
        if  reward[-1]>=BUY_GATE and reward[-2]>=BUY_GATE:
            code_4_buy.append({'code':stock,
                    'price':item['time_close'][0][1],'reward':reward[-1]})

        if (reward[-1]<=SELL_GATE and reward[-2]<=SELL_GATE):
            code_4_sell.append({'code':stock,
                    'price':item['time_close'][0][1],'reward':reward[-1]})

    g_market_train_data = 0
    g_normalized_data = 0
    g_max_step = 0
    train_x = 0
    train_y = 0

    #sess.close()

    if len(code_4_buy)>0:
        #字典列表按键值‘a’逆序排序  a.sort(key=lambda x:-x['a']
        code_4_buy.sort(key=lambda x:x['reward'],reverse=True)

        #取涨幅最大的前五
        buy_list=code_4_buy[:MAX_HOLDING]
        now='[{0}]'.format( time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        print(now,stop_dt,' buy list:', {x['code'] for x in buy_list})
    else:
        buy_list=[]

    if len(code_4_sell)>0:
        code_4_sell.sort(key=lambda x: x['reward'], reverse=False)
        sell_list = code_4_sell
        now = '[{0}]'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(now,stop_dt,' sell list:', {x['code'] for x in sell_list[:6]})
    else:
        sell_list=[]

    if len(code_4_sell)==0 and len(code_4_buy)==0:
        now = '[{0}]'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print(now,stop_dt, ' no trade')

    return   buy_list,sell_list

def get_stock_list():
    return g_test_securities

g_test_securities=gmTools.get_index_stock(STOCK_BLOCK)[:MAX_STOCKS]
#study()
if __name__ == '__main__':
    #训练模型
    train_model()
else:
    #回测期间不能反复初始化模型，否则会导致运行越来越慢
    model_path = g_log_dir

    sess = tf.InteractiveSession()
    setup_tensor(sess, '', g_week, 0)
    saver = tf.train.Saver()

    model_file = tf.train.latest_checkpoint(model_path)

    if model_file:
        try:
            saver.restore(sess, model_file)
            week, code = sess.run([model_week, model_code])
            # code string ,返回是bytes类型，需要转换
            print("restore from model code=%s,week=%d" % (
                code, week))
        except:
            print("restore from model error" )
            pass
