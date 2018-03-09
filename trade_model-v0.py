# coding=utf-8

import tensorflow as tf
import numpy  as np
import time
import gmTools
import struct

""" 
2018-02-25:
    1）在基本功能的基础上，增加基于特定stock的模型管理，图中增加用于记录本模型适应的stock代码、
    训练过的时间范围、频率、模型平均准确率等信息，后续使用时先判断是否有可用的模型，如果有即加载
    模型后继续训练或直接使用；
    2）模型应能持续优化，使用时间越长、精度应该越高；
    3)无法解决utc时间到字符串的转后的结果在gm读取信息中的使用问题，int(week*60)强制类型转换
    4)同一计算图在一次运行时不能多次生成、使用

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


"""

g_max_step = 20000
g_learning_rate = 0.001
g_dropout = 0.9

# 策略参数
g_week = 120
g_max_holding_days = 15

g_input_columns = 6
g_trade_minutes = 240
g_week_in_trade_day = int(g_trade_minutes / g_week)
g_look_back_weeks = 40  # 回溯分析的周期数
g_max_holding_weeks = g_week_in_trade_day * g_max_holding_days  # 用于生成持仓周期内的收益等级
g_max_stage = 11  # 持仓周期内收益等级
g_stage_rate = 2  # 持仓周期内收益等级差

g_log_dir = 'logs/week{0}hold{1}days'.format(g_week, g_max_holding_days)

g_train_stop = 0  # 当前测试数据结束位置
g_test_stop = 0  # 当前实时数据结束位置

g_train_startDT = '2017-01-01'  # oldest start 2015-01-01
g_train_stopDT = '2018-12-15'

g_trade_startTime = ' 09:30:00'
g_trade_stopTime = ' 15:00:10'

g_test_securities = ["SZSE.002415","SZSE.000333","SZSE.002460"]


# 标签数据生成，自动转成行向量 0-10共11级，级差1%
def make_stage(x):
    x = int(100 * x / g_stage_rate)
    if abs(x) < 5:
        x = x + 5
    elif x > 4:
        x = 10
    else:
        x = 0

    tmp = np.zeros(g_max_stage, dtype=np.int)
    tmp[x] = 1
    return tmp


# 获取测试集  步进为time_step的测试数据块,
# 与训练数据格式不一致、处理方式也不一致，预测周期越长越不准确
# 数据块不完整时必须用0补充完整
def get_test_data(data, normalized_data, look_back_weeks=g_look_back_weeks):
    test_x, test_y = [], []

    for i in range(look_back_weeks):
        test_x.append([0])
        test_y.append([0])

    for i in range(look_back_weeks, len(data)):
        x = normalized_data.iloc[i - look_back_weeks:i, :]
        y = data.iloc[i - look_back_weeks:i, -1]

        test_x.append(x.values.tolist())

        # test_y.extend(y.values.tolist())
        test_y.append(y.values.tolist())

    return test_x, test_y


def create_market_data(stock, start_DateTime, stop_DateTime,
                       week=g_week, look_back_weeks=g_look_back_weeks, start_utc=0):
    global g_market_train_data, g_input_columns, \
        g_normalized_data, g_max_step, train_x, train_y

    g_market_train_data = gmTools.read_kline(stock, int(week * 60),
                                             start_DateTime, stop_DateTime)  # 训练数据
    g_market_train_data['label'] = g_market_train_data['close'].pct_change(g_max_holding_weeks)
    # 将数据总项数整理成g_max_holding_weeks的整数倍
    # tmp = len(g_market_train_data)%g_max_holding_weeks+g_max_holding_weeks
    # g_market_train_data =g_market_train_data[tmp:]

    g_market_train_data['label'] = g_market_train_data['label'].shift(-1).fillna(0)
    g_market_train_data['label'] = g_market_train_data['label'].apply(make_stage)
    data_tmp = g_market_train_data.iloc[:, 1:-1]
    # todo  加入其他的技术分析指标

    # 数据归一化处理

    mean = np.mean(data_tmp, axis=0)
    std = np.std(data_tmp, axis=0)
    g_normalized_data = (data_tmp - mean) / std  # 标准化

    g_input_columns = len(data_tmp.columns)

    cols = ['endtime', 'close', 'label']
    g_market_train_data = g_market_train_data[cols]
    g_max_step = len(g_market_train_data)

    train_x, train_y = get_test_data(g_market_train_data,
            g_normalized_data, look_back_weeks)


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
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
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
             output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weight'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)

        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)

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

g_nn_hidden_nodes = 0
hidden1 = 0
x = 0
y = 0
y1 = 0
cross_entropy = 0
train_step = 0
buy = 0
sell = 0
accuracy = 0
merged = 0
train_writer = 0
test_writer = 0
keep_prob = 0
sell_prediction = 0
buy_prediction = 0
valid_accuracy = 0
valid_accuracy2 = 0
model_code = 0
model_last_utc = 0
model_next_train_utc = 0
model_week = 0


def setup_tensor(sess, stock, week, last_utc, next_train_time=0):
    global g_nn_hidden_nodes, hidden1, cross_entropy, \
        train_step, buy, sell, accuracy, merged, train_writer, \
        test_writer, keep_prob, x, y, y1, sell_prediction, \
        buy_prediction, valid_accuracy, valid_accuracy2, \
        model_code, model_last_utc, model_week, model_next_train_utc

    """ 
    为了在TensorBoard中展示节点名称，设计网络时会常使用tf.name_scope限制命名空间， 
    在这个with下所有的节点都会自动命名为input/xxx这样的格式。 
    定义输入x和y的placeholder，并将输入的一维数据变形为28×28的图片存储到另一个tensor， 
    这样就可以使用tf.summary.image将图片数据汇总给TensorBoard展示了。 
    """

    # 定义网络参数
    with tf.name_scope("model_vars"):
        model_code = tf.Variable(stock, dtype=tf.string, trainable=False, name="model_code")
        model_week = tf.Variable(week, dtype=tf.int32, trainable=False, name="model_week")
        model_last_utc = tf.Variable(last_utc, dtype=tf.int64,
                                     trainable=False, name="model_last_utc")
        model_next_train_utc = tf.Variable(next_train_time, dtype=tf.int64,
                                           trainable=False, name="model_next_train_utc")

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None,
                                        1 * g_input_columns], name='x_input')
        y = tf.placeholder(tf.int32, [None, g_max_stage], name='y_input')

    with tf.name_scope('input_reshape'):
        x_shaped_input = tf.reshape(x,
                                    [-1, 1, g_input_columns, 1])
        tf.summary.image('input', x_shaped_input, 1)

    g_nn_hidden_nodes = 500
    hidden1 = nn_layer(x, 1 * g_input_columns, g_nn_hidden_nodes, 'layer1')

    with tf.name_scope('g_dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y1 = nn_layer(dropped, g_nn_hidden_nodes, g_max_stage, 'layer2', act=tf.identity)

    """ 
    这里使用tf.nn.softmax_cross_entropy_with_logits()
    对前面输出层的结果进行softmax处理并计算交叉熵损失cross_entropy。 
    计算平均损失，并使用tf.summary.saclar进行统计汇总。 
    """
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(logits=y1, labels=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
            tf.summary.scalar('cross_entropy', cross_entropy)

    """ 
    使用Adma优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuray， 
    再使用tf.summary.scalar对accuracy进行统计汇总。 
    """
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(g_learning_rate).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('buy_prediction'):
            buy_prediction = tf.greater_equal(tf.argmax(y1, 1), 6)
            buy = tf.reduce_max(tf.cast(buy_prediction, tf.float32))
            tf.summary.scalar('buy', buy)

        with tf.name_scope('sell_prediction'):
            sell_prediction = tf.less_equal(tf.argmax(y1, 1), 3)
            sell = tf.reduce_max(tf.cast(sell_prediction, tf.float32))
            tf.summary.scalar('sell', sell)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y1, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        with tf.name_scope('valid_accuracy2'):
            valid_prediction2 = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), 2 * g_stage_rate)
            valid_accuracy2 = tf.reduce_mean(tf.cast(valid_prediction2, tf.float32))
            tf.summary.scalar('valid_accuracy2', valid_accuracy2)

        with tf.name_scope('valid_accuracy'):
            valid_prediction = tf.less_equal(abs(tf.argmax(y1, 1) - tf.argmax(y, 1)), g_stage_rate)
            valid_accuracy = tf.reduce_mean(tf.cast(valid_prediction, tf.float32))
            tf.summary.scalar('valid_accuracy', valid_accuracy)
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
    train_writer = tf.summary.FileWriter(g_log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(g_log_dir + '/test')
    tf.global_variables_initializer().run()


def feed_dict(train):
    global g_train_stop

    # todo  数据取完后如何处理？  直接退出运行,需支持对未来收益的预测
    if train:
        k = g_dropout
    else:
        k = 1.0

    try:
        xs = train_x[g_train_stop]
        ys = train_y[g_train_stop]
        g_train_stop += 1
    except:
        xs = 0
        ys = 0
        pass

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


def main(stock=g_test_securities, week=g_week, look_back_weeks=g_look_back_weeks):
    global g_train_startDT, g_train_stop, g_market_train_data, \
        g_normalized_data, g_max_step, train_x, train_y

    # model_code, model_last_utc, model_week
    has_bought = False
    buy_count = 0
    buy_week = 0
    buy_ok = 0
    buy_price = 0
    buy_info = ""
    output_count = 0
    sess = tf.InteractiveSession()
    setup_tensor(sess, stock, g_week, 0)
    saver = tf.train.Saver(max_to_keep=3)

    g_test_securities = gmTools.get_index_stock('SZSE.399300')

    # try:
    for stock in g_test_securities:
        model_path = g_log_dir + "/" + stock
        g_train_stop = max(look_back_weeks, g_max_holding_weeks)
        model_file = tf.train.latest_checkpoint(model_path)

        model_valid = False
        if model_file:
            try:
                saver.restore(sess, model_file)
                week, code = sess.run([model_week, model_code])
                # code string ,返回是bytes类型，需要转换
                print("restore from model code=%s,week=%d" % (code, week))
                model_valid = True
            except:
                pass

        if model_valid:
            startDT = time.localtime(int(sess.run(model_next_train_utc)))
            startDT = time.strftime('%Y-%m-%d %H:%M:%S', startDT)
        else:
            startDT = g_train_startDT + g_trade_startTime

        create_market_data(stock=stock,
                           start_DateTime=startDT,
                           stop_DateTime=g_train_stopDT + g_trade_stopTime,
                           week=week, look_back_weeks=g_look_back_weeks)

        g_max_step = len(g_market_train_data)
        next_train_utc = g_max_step - g_look_back_weeks - 1 - g_week_in_trade_day * 4
        if next_train_utc < 0:
            i = 0
            return

        # 记录模型目前训练终止时间与下次训练开始时间
        # TODO 回退分析时数据对不齐，有时重叠、有时错位
        # TODO 数据项太少时容易出现无法进行后续分析的情况，需要考虑加大最小数据项的获取值
        sess.run(model_next_train_utc.assign(
            g_market_train_data['endtime'][next_train_utc]))

        sess.run(model_last_utc.assign(
            g_market_train_data['endtime'][g_max_step - 1]))

        print('log dir:%s , total items :%d' % (g_log_dir, g_max_step))
        begin_buy_sell_analyse = int(g_max_step * 0.6)
        for i in range(look_back_weeks, g_max_step):
            try:
                if i % 100 == 0:
                    summary, acc, valid_acc, valid_acc2 = sess.run([merged,
                        accuracy, valid_accuracy, valid_accuracy2],
                        feed_dict=feed_dict(False))
                    test_writer.add_summary(summary, i)
                    # print('--{0}--at[{1}][{2}%] acc[{3}]valid-acc[{4}]valid-acc2[{5}]'.format(
                    #    g_log_dir,i,int(i*100/g_max_step),acc,valid_acc,valid_acc2))
                elif i % 200 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                          options=run_options, run_metadata=run_metadata)

                    # ????
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)

                    # print('Adding run metadata for', i)
                else:
                    summary, is_sell, is_buy, _ = sess.run([merged, sell_prediction,
                            buy_prediction, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)

                    # 训练一段时间后才进行买卖点分析
                    if i > begin_buy_sell_analyse:
                        if not has_bought and is_buy[-1] and is_buy[-2]:
                            buy_count += 1
                            output_count = i
                            buy_week = i
                            has_bought = True
                            buy_price = g_market_train_data['close'][i]
                            buy_info = ("[%s] [%s]buy [%03.2f]"
                                        % (gmTools.timestamp_datetime(g_market_train_data['endtime'][i]),
                                           stock, g_market_train_data['close'][i]))

                        if (has_bought and i - buy_week > g_week_in_trade_day):
                            if ((is_sell[-1] and is_sell[-2])
                                or (i - buy_week > g_max_holding_weeks)
                                or g_market_train_data['close'][i] / buy_price < 0.9):

                                has_bought = False
                                reward = int(g_market_train_data['close'][i] * 100 / buy_price) - 100
                                if reward > -1:
                                    # 有盈利的操作
                                    buy_ok += 1
                                output_count = i
                                print("%s--[%03.2f] reward:%03d sell [%s] "
                                      % (buy_info, g_market_train_data['close'][i], reward,
                                         gmTools.timestamp_datetime(g_market_train_data['endtime'][i])))
                                buy_info = ""
            except:
                print("error  in main")
                pass

            if i - output_count > int(begin_buy_sell_analyse / 3):
                # 防止长时间计算无输出
                output_count = i
                print("analysing %s" % (gmTools.timestamp_datetime(g_market_train_data['endtime'][i])))

        i = len(g_market_train_data) - 1
        if buy_count > 0:
            if len(buy_info) > 1:
                # the lastest buy
                reward = int(g_market_train_data['close'][i] * 100 / buy_price) - 100
                if reward > 0:
                    # 有盈利的操作
                    buy_ok += 1

                print("%s--[%03.2f] reward:%03d sell [%s] "
                      % (buy_info, g_market_train_data['close'][i], reward,
                         gmTools.timestamp_datetime(g_market_train_data['endtime'][i])))

            print("total buy:%d,sucess:%.2f%%,buy_ok =%d" % (buy_count, buy_ok * 100 / buy_count, buy_ok))
        else:
            print("no trade ")
        if i > 0:
            print("end in [%s]\n" % (gmTools.timestamp_datetime(g_market_train_data['endtime'][i])))
        else:
            print("end no datas\n")
        '''
            except:
            pass
        finally:
        '''
        i = len(g_market_train_data) - 1
        if i > 0:
            sess.run(model_last_utc.assign(g_market_train_data['endtime'][i]))
            saver.save(sess, model_path + "/model.ckpt")  # save模型

        '''
        test_writer.flush()
        train_writer.flush()
        test_writer.close()
        train_writer.close()

        sess.close()
        
        # 返回最后终止的时间,便于调用者确定是否需要继续调用下一次分析
        if i > 0:
            last_utc = g_market_train_data['endtime'][i]
        else:
            last_utc = 0

        g_market_train_data = 0
        g_normalized_data = 0
        g_max_step = 0
        train_x = 0
        train_y = 0

        return last_utc
        '''

'''
for stock in g_test_securities:
    print("\n[%s]start analysing [%s]" % (stock,
                                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    main(stock=stock, week=g_week, look_back_weeks=g_look_back_weeks)
'''

main()