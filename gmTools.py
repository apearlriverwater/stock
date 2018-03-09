#-*- coding: utf-8 -*-
 
import struct
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy  as np
import  matplotlib.pyplot as plt
import datetime
from datetime import datetime
import time
from gmsdk.enums import *
from gmsdk import td,md, get_strerror, to_dict
import  talib as ta
import tushare as ts

RISINGSUN_BUY	=		1  #BUY 
RISINGSUN_SELL	=		2  #SELL

#utc 时间戳转换
def timestamp_datetime(ts):
    if isinstance(ts, (int, np.int64,float, str)):
        try:
            ts = int(ts)
        except ValueError:
            raise

        if len(str(ts)) == 13:
            ts = int(ts / 1000)
        if len(str(ts)) != 10:
            raise ValueError
    else:
        raise ValueError()

    return datetime.fromtimestamp(ts)


def datetime_timestamp(dt, type='ms'):
    if isinstance(dt, str):
        try:
            if len(dt) == 10:
                dt = datetime.strptime(dt.replace('/', '-'), '%Y-%m-%d')
            elif len(dt) == 19:
                dt = datetime.strptime(dt.replace('/', '-'), '%Y-%m-%d %H:%M:%S')
            else:
                raise ValueError()
        except ValueError as e:
            raise ValueError(
                "{0} is not supported datetime format." \
                "dt Format example: 'yyyy-mm-dd' or yyyy-mm-dd HH:MM:SS".format(dt)
            )

    if isinstance(dt, time.struct_time):
        dt = datetime.strptime(time.stftime('%Y-%m-%d %H:%M:%S', dt), '%Y-%m-%d %H:%M:%S')

    if isinstance(dt, datetime):
        if type == 'ms':
            ts = int(dt.timestamp()) * 1000
        else:
            ts = int(dt.timestamp())
    else:
        raise ValueError(
            "dt type not supported. dt Format example: 'yyyy-mm-dd' or yyyy-mm-dd HH:MM:SS"
        )
    return ts

'''
path: 
filename:文件名包含的子字符串，不支持通配符
onlyfile=True  是否仅返回文件名，不返回子目录名
'''
def get_code_in_cap_file(path,filename,minutes,onlyfile=True):
    lists=os.listdir(path)
    files=[]

    if onlyfile==True:
        #only return file lists
        if len(filename)>0:
            files=[file for file in lists if file.find(filename)>-1 and file.find('.dat')>-1 and file.find(minutes)>-1]
            return files

    return lists

#//仅保留有用的信息
#typedef struct tagCAPITALFLOWMINISTRUCK {
#	int32_t	m_nDate, m_nTime;       //date /时间  2*4
#	double	m_dblSmallBuy, m_dblMidBuy, m_dblBigBuy, m_dblHugeBuy;   4*8
#	double	m_dblSmallSell, m_dblMidSell, m_dblBigSell, m_dblHugeSell;  4*8

def read_cap_flow(filepath):
    columns=['Date','Time','SmallBuy','MidBuy','BigBuy','HugeBuy','SmallSell','MidSell','BigSell','HugeSell']

    f = open(filepath,'rb')
    dataSize=72
    filedata = f.read()
    filesize = f.tell()
    f.close()
    
    tickCount=filesize/dataSize

    index=0
    series=[]
    while index < filesize:
        cap=struct.unpack_from('2i8d',filedata,index)
        series.append(cap)
        index=index+dataSize
    caps=pd.DataFrame(series,columns=columns)
    return caps


'''
    主力资金流统计
'''
def CaclMainFlow(CapFlow):
    MainFlow=()
    for flow in CapFlow:
        MainFlow=MainFlow+((MainFlow+flow[4]+flow[5]-flow[8]-flow[9])/10000)
        continue
    return MainFlow  #单位：万元  list

'''
    均线多头向上判断：多头向上时返回true，否则false
        dataList 待计算数据
        maList 周期序列列表，最少三个周期,
        nLastWeeks最少程序周期数
'''
def IsMaUp(data,maList,nLastWeeks):
    bRet=True
    ma=[]
    columns=[]
    CaclCount=sum(maList)+nLastWeeks+2

    if len(maList)>=3 and len(data)>CaclCount:
         # 计算每个周期的主力资金流变化情况
        mainflow=data['BigBuy']+data['HugeBuy']-data['BigSell']-data['HugeSell']


        #分析资金流变化情况的均线趋势  按列排序并进行比较
        for week in maList:
            columns.append(str(week))
            tmp=mainflow.rolling(week).mean().tolist()
            ma.append(tmp[-nLastWeeks-1:])

        #分析资金流变化情况的均线趋势  按列排序并进行比较
        index=0
        while index<len(maList):
            #按列进行排序，分析各点数据  大到小排序
            tmp=ma[index]
            if tmp!=sorted(tmp,reverse = True):
                bRet=False
                break

            index=index+1

    else:
        bRet=False

    return bRet

'''
    均线多头向下判断：多头向下时返回true，否则false
        dataList 待计算数据
        maList 周期序列列表，最少三个周期,
        nLastWeeks最少程序周期数
'''
def IsMaDown(data,maList,nLastWeeks):
    bRet=True
    ma=[]
    columns=[]
    CaclCount=sum(maList)+nLastWeeks+2

    if len(maList)>=3 and len(data)>CaclCount:
         # 计算每个周期的主力资金流变化情况
        #data['mainflow']=data['BigBuy']+data['HugeBuy']-data['BigSell']-data['HugeSell']
        mainflow=data['BigBuy']+data['HugeBuy']-data['BigSell']-data['HugeSell']
        #mainflow=data['BigBuy']+data['HugeBuy']-data['BigSell']-data['HugeSell']
        for week in maList:
            columns.append(str(week))
            tmp=mainflow.rolling(week).mean().tolist()
            ma.append(tmp[-nLastWeeks-1:])

        #分析资金流变化情况的均线趋势  按列排序并进行比较
        #pdMa=pd.DataFrame(ma,index=columns)
        index=0
        while index<len(maList):
            #按列进行排序，分析各点数据  大到小排序
            tmp=ma[index]
            if tmp!=sorted(tmp,reverse = False):
                bRet=False
                break;

            index=index+1
    else:
        bRet=False

    return bRet


#//必须固定为17字节数据，采用结构体单字节对齐方式
#typedef struct tagL2TICKS {
#	int m_nTime, m_nIndex;       //时间、成交笔序号
#	int m_nPriceMul1000, m_nVols;//价格*1000，成交股数
#	char m_nBS;                  //成交方向：2买  1卖 0 竞价？
#}L2TICKS;    
#nTime,nIndex,nPrice1000,nVol,cBS
def read_ticks(tickfilepath):
    columns=['Time','Index','PriceMul1000','Vol','BS']
    f = open(tickfilepath,'rb')
    filedata = f.read()
    filesize = f.tell()
    f.close()
    dataSize=17
    tickCount=filesize/dataSize

    index=0
    series=[]

    while index < filesize:
        tick=struct.unpack_from('4i1c',filedata,index)
        series.append(tick)
        index=index+dataSize

    ticks=pd.DataFrame(series,columns=columns)
    return ticks


'''
    利用掘金终端的函数读取指数的成份股
    stock_list :"SHSE.600000,SZSE.000001"
'''
def get_index_stock(index_symbol,return_list=True):
    # 连接本地终端时，td_addr为localhost:8001,
    if (td.init('haigezyj@qq.com', 'zyj2590@1109', 'strategy_1') == 0):
        try:
            stock_list=""
            css=md.get_constituents(index_symbol)

            if return_list:
                stock_list=[ cs.symbol for cs in css]
            else:
                for cs in css:
                    stock_list +="," +cs.symbol
            return stock_list[1:]
        except:
            pass

'''
    利用掘金终端的函数读取各市场的可交易标的
    exchange:
        上交所，市场代码 SHSE
        深交所，市场代码 SZSE
        中金所，市场代码 CFFEX
        上期所，市场代码 SHFE
        大商所，市场代码 DCE
        郑商所，市场代码 CZCE
        纽约商品交易所， 市场代码 CMX (GLN, SLN)
        伦敦国际石油交易所， 市场代码 IPE (OIL, GAL)
        纽约商业交易所， 市场代码 NYM (CON, HON)
        芝加哥商品期货交易所，市场代码 CBT (SOC, SBC, SMC, CRC)
        纽约期货交易所，市场代码 NYB (SGN)
    sec_type 	int 	代码类型:1 股票，2 基金，3 指数，4 期货，5 ETF
    is_active 	int 	当天是否交易：1 是，0 否
    
    Instrument
        交易代码数据类型
        class Instrument(object):
            def __init__(self):
                self.symbol = ''                ## 交易代码
                self.sec_type = 0               ## 代码类型
                self.sec_name = ''              ## 代码名称
                self.multiplier = 0.0           ## 合约乘数
                self.margin_ratio = 0.0         ## 保证金比率
                self.price_tick = 0.0           ## 价格最小变动单位
                self.upper_limit = 0.0          ## 当天涨停板
                self.lower_limit = 0.0          ## 当天跌停板
                self.is_active = 0              ## 当天是否交易
                self.update_time = ''           ## 更新时间

    
    stock_list :"SHSE.600000,SZSE.000001"
'''
def get_stock_by_market(exchange,sec_type,is_active,return_list=True):
    # 连接本地终端时，td_addr为localhost:8001,
    if (td.init('haigezyj@qq.com', 'zyj2590@1109', 'strategy_1') == 0):
        try:
            stock_list=""
            css=md.get_instruments(exchange, sec_type, is_active)

            if return_list:
                stock_list=[ cs.symbol for cs in css]
            else:
                for cs in css:
                    stock_list +="," +cs.symbol
            return stock_list[1:]
        except:
            pass
'''
    利用掘金终端的函数读取指定股票最新价，用于统计当日当时价位情况
    stock_list :"SHSE.600000,SZSE.000001"
'''
def get_minutes_bars(stock_list,minutes,begin_time, end_time):
    # 连接本地终端时，td_addr为localhost:8001,
    if (td.init('haigezyj@qq.com', 'zyj2590@1109', 'strategy_1') == 0):
        try:
            bars = md.get_bars(stock_list, int(minutes*60),begin_time, end_time)
            return bars
        except:
            pass

def get_daily_bars(stock_list,begin_time, end_time):
    # 连接本地终端时，td_addr为localhost:8001,
    if (td.init('haigezyj@qq.com', 'zyj2590@1109', 'strategy_1') == 0):
        try:
            bars = md.get_dailybars(stock_list,begin_time, end_time)
            return bars
        except:
            pass
'''
利用掘金终端的函数读取需要的K线数据
get_bars 提取指定时间段的历史Bar数据，支持单个代码提取或多个代码组合提取。策略类和行情服务类都提供该接口。
get_bars(symbol_list, bar_type, begin_time, end_time)
        参数名	类型	说明
        symbol_list	string	证券代码, 带交易所代码以确保唯一，如SHSE.600000，同时支持多只代码
        bar_type	int	bar周期，以秒为单位，比如60即1分钟bar
        begin_time	string	开始时间, 如2015-10-30 09:30:00
        end_time	string	结束时间, 如2015-10-30 15:00:00
return:dataframe  'endtime','open','high','low','close','volume','amount'
'''
def read_kline(symbol_list, weeks_in_seconds, begin_time, end_time,max_record=50000):
    
    #连接本地终端时，td_addr为localhost:8001,
    if(td.init('haigezyj@qq.com', 'zyj2590@1109', 'strategy_1')==0):
        #类结构体转成dataframe
        kdata=[]
        columns=['endtime','open','high','low','close','volume','amount']
        bars=0

        is_daily=(weeks_in_seconds==240*60)

        while (True):

            # 返回结果是bar类数组
            if is_daily:
                bars = md.get_dailybars(symbol_list, begin_time, end_time)
            else:
                bars = md.get_bars(symbol_list, weeks_in_seconds, begin_time, end_time)

            for bar in bars:
                if is_daily:
                    kdata.append([int(bar.utc_time),
                        bar.open, bar.high, bar.close, bar.low,
                        bar.volume, bar.amount])
                else:
                    kdata.append([int(bar.utc_endtime),
                        bar.open,bar.high,bar.close,bar.low,
                        bar.volume,bar.amount])

            count=len(bars)
            #TODO 一次最多处理10000项以内数据，超出应有所提示
            if (count==0 or len(kdata)>max_record) \
               or (not is_daily and bars[count - 1].strendtime >= end_time) \
               or (is_daily and bars[count - 1].strtime >= end_time)   :
                break

            #print("read [%s] k line:%s count=%d" % (symbol_list,
            #        bars[0].strtime[:10] + ' ' + bars[0].strtime[11:19], count))

            if is_daily:
                if count<=10:
                    break
                else:
                    begin_time = bars[count - 1].strtime[:10] \
                                 + ' ' + bars[count - 1].strtime[11:19]
            else:
                begin_time=bars[count-1].strendtime[:10]\
                           +' '+bars[count-1].strendtime[11:19]
        '''
        count=len(kdata)
        if count>0:
            print("total count:%d,end date:%s " % (count,timestamp_datetime(kdata[count-1][0])))
        else:
            print("No data" )
        '''

        return pd.DataFrame(kdata,columns=columns)


def read_kline_ts(symbol_list, weeks_in_seconds, begin_time, end_time, max_record=50000):
    if (True):
        # 类结构体转成dataframe
        kdata = []
        columns = ['endtime', 'open', 'high', 'low', 'close', 'volume', 'amount']
        bars = 0

        is_daily = (weeks_in_seconds == 240 * 60)

        while (True):

            # 返回结果是bar类数组
            if is_daily:
                bars = md.get_dailybars(symbol_list, begin_time, end_time)
            else:
                bars = md.get_bars(symbol_list, weeks_in_seconds, begin_time, end_time)

            for bar in bars:
                if is_daily:
                    kdata.append([int(bar.utc_time),
                                  bar.open, bar.high, bar.close, bar.low,
                                  bar.volume, bar.amount])
                else:
                    kdata.append([int(bar.utc_endtime),
                                  bar.open, bar.high, bar.close, bar.low,
                                  bar.volume, bar.amount])

            count = len(bars)
            # TODO 一次最多处理10000项以内数据，超出应有所提示
            if (count == 0 or len(kdata) > max_record) \
                    or (not is_daily and bars[count - 1].strendtime >= end_time) \
                    or (is_daily and bars[count - 1].strtime >= end_time):
                break

            # print("read [%s] k line:%s count=%d" % (symbol_list,
            #        bars[0].strtime[:10] + ' ' + bars[0].strtime[11:19], count))

            if is_daily:
                if count <= 10:
                    break
                else:
                    begin_time = bars[count - 1].strtime[:10] \
                                 + ' ' + bars[count - 1].strtime[11:19]
            else:
                begin_time = bars[count - 1].strendtime[:10] \
                             + ' ' + bars[count - 1].strendtime[11:19]
        '''
        count=len(kdata)
        if count>0:
            print("total count:%d,end date:%s " % (count,timestamp_datetime(kdata[count-1][0])))
        else:
            print("No data" )
        '''

        return pd.DataFrame(kdata, columns=columns)

def read_last_n_kline(symbol_list, weeks_in_seconds, count, end_time):
    # 连接本地终端时，td_addr为localhost:8001,
    if (td.init('haigezyj@qq.com', 'zyj2590@1109', 'strategy_1') == 0):
        # 类结构体转成dataframe
        columns = ['endtime', 'open', 'high', 'low', 'close', 'volume', 'amount']
        bars = 0

        is_daily = (weeks_in_seconds == 240 * 60)
        data_list =[] # pd.DataFrame(None, columns=columns)
        '''
        todo 整批股票读取有问题，数据取不全，放弃
        stocks = ''
        for x in symbol_list:
            stocks+=','+x

        read_days=int(count*weeks_in_seconds/240/60)+1
        start_date=md.get_calendar('SZSE',
            datetime.datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                -datetime.timedelta(days=read_days),end_time)[0].strtime
        start_date=start_date[:10] +' 09:30:00'

        while start_date<end_time:
            bars=md.get_bars(stocks[1:], weeks_in_seconds, start_date, end_time)
        '''
        for stock in symbol_list:
            #now = '[{0}] read k line'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            #print(now,stock)
            kdata = []
            # 返回结果是bar类数组
            if is_daily:
                bars = md.get_last_n_dailybars(stock, count, end_time)
            else:
                bars = md.get_last_n_bars(stock, weeks_in_seconds, count, end_time)

            for bar in bars:
                if is_daily:
                    kdata.append([int(bar.utc_time),
                                  bar.open, bar.high, bar.low, bar.close,
                                  bar.volume, bar.amount])
                else:
                    kdata.append([int(bar.utc_endtime),
                                  bar.open, bar.high, bar.low, bar.close,
                                  bar.volume, bar.amount])


            if len(bars)>0:
               kdata=pd.DataFrame(kdata, columns=columns)
               kdata=kdata.sort_values(by='endtime',ascending=False)
               data_list.append({'code':stock,'kdata':kdata})

        return data_list
'''
图形化显示标的走势
'''
def draw_figure(data1,data2=None,title=''):
    # 以折线图表示结果 figsize=(20, 15)
    plt.figure()
    plot_predict = plt.plot(list(range(len(data1))),
                            data1, color='b', label='predict')
    if data2!=None:
        plot_test_y = plt.plot(list(range(len(data2))),
                               data2, color='r', label='true')

    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')

    if len(title)>0:
        plt.title(title)

    #plt.show()

    return plt

def show_BS(plt,point,price,is_buy=True,title=''):
    if is_buy:
        plt.annotate('b', xy=(point, price),
                     xytext=(point * 1.1, price),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     )
    else:
        plt.annotate('s', xy=(point, price),
                     xytext=(point * 0.9, price),
                     arrowprops=dict(facecolor='green', shrink=0.05),
                     )

    if len(title)>0:
        plt.title(title)