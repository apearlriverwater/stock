# -*- coding: utf-8 -*-
#利用本框架进行ai模型测试

from gmsdk.api import StrategyBase
from gmsdk import get_strerror
import trade_model_v3 as tm
import time
import gmTools

eps = 1

class Mystrategy(StrategyBase):
    def __init__(self, *args, **kwargs):
        super(Mystrategy, self).__init__(*args, **kwargs)
        self.buy_list=[]  #字典列表
        self.sell_list=[]
        self.trade_count=0
        self.trade_limit=5
        self.utc_endtime=0
        self.now=''
        self.positions =''
        self.holding_list=[]
        self.sells=''

    #TODO 考虑利用自定义定时器事件或者交易所事件在每个交易日结束后自动进行模型训练完善
    # 事件未发生，原因待查
    def on_md_event(self,md_event):
        print('on_md_event ',md_event)
        pass


    def on_bar(self, bar):
        '''
        self.buy_list = [{'code': 'SZSE.000001', 'price': 12.34, 'reward': 5},
                         {'code': 'SHSE.600000', 'price': 32.34, 'reward': 5}]
        self.sell_list=[{'code':'SZSE.000001','price':12.34,'reward':5},
                       {'code':'SHSE.600000','price':32.34,'reward':5}]
        '''

        if self.utc_endtime!=bar.utc_endtime:
            print("[%s]on_bar " % (time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))))

            self.positions = self.get_positions()  # 查询策略所持有的多仓
            # 打印持仓信息
            # self.print_position()
            self.sells = [x['code'] for x in self.sell_list]

            stop_time = bar.strtime[:10] + ' ' + bar.strtime[11:19]
            self.utc_endtime=bar.utc_endtime
            self.now=stop_time
            #一次读取多日或多周期的K线数据进行分析，提交整体运行效率，
            # 买卖处理也采用自行定义方式，掘金量化仅提供所需的基础数据
            self.buy_list,self.sell_list=tm.get_bs_list(stop_time)

            # 打印持仓信息
            self.print_position()

        self.trade(bar.exchange+ '.'+ bar.sec_id,bar)
        pass

    #TODO  持仓股价沟通完毕才打印持仓信息
    def print_position(self):
        positions = self.get_positions()  # 查询策略所持有的多仓
        if  len(positions) > 0:
            now = '[{0}]'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            print(now,self.now,'holding ...')
            holding=0
            for position in positions:
                stock=position.exchange+'.'+ position.sec_id
                self.holding_list.append(stock)
                price=self.get_last_n_bars(stock,60,1,self.now)[0].close
                reward = price * 100 / position.vwap-100
                tmp=position.volume*price
                print("     code:%s,vol:%d,amount:%.2f,reward:%.2f" % (
                    stock, position.volume,tmp , reward))
                holding+=tmp
            cash=self.get_cash()
            print("     nav:%.2f,av:%.2f,hold:%.2f,reward:%.2f%%\n"
                  % (cash.nav,cash.available,holding,cash.profit_ratio ))

    #交易处理有关函数
    def trade(self,sec,bar):
        if sec in self.holding_list:
             for position in self.positions:
                stock = position.exchange + '.' + position.sec_id
                if  stock==sec:
                    if position.available_yesterday > eps :
                        if stock in self.sells :
                            # sell out stock in sell list
                            self.close_long(position.exchange, position.sec_id,
                                            0, position.available_yesterday)
                        else:
                            price = self.get_last_n_bars(stock, 60, 1, self.now)[0].close
                            reward = price * 100 / position.vwap - 100
                            if reward<-8:  #stop loss
                                # sell out
                                self.close_long(position.exchange, position.sec_id,
                                                0, position.available_yesterday)

                    break

        if len(self.buy_list)>0:
            for i in  range(len(self.buy_list)):
                item=self.buy_list[i]
                stock = item['code']

                # 没有超出下单次数限制
                if stock == sec:
                    if len(self.positions) < self.trade_limit :
                        #a stock only can buy once
                        holding=False
                        for position in self.positions:
                            if sec==position.exchange+'.'+position.sec_id:
                                holding=True
                                break

                        if holding==False:
                            price = bar.close
                            cash=self.get_cash()
                            vol = int(cash.available/
                                    ((self.trade_limit-len(self.positions) )*price*130))
                            # 如果本次下单量大于0,  发出买入委托交易指令
                            if vol >= eps:
                                order=self.open_long(stock[:4], stock[5:], price, vol*100)

                    break
                    
if __name__ == '__main__':
    myStrategy = Mystrategy(
        username='haigezyj@qq.com',
        password='zyj2590@1109',
        strategy_id='31dbb817-1c57-11e8-b832-dc5360304926',
        mode=4
        #,td_addr='127.0.0.1:8001'
    )
    myStrategy.backtest_config(
        start_time='2017-07-01 09:30:00',
        end_time='2018-12-11 15:20:00',
        initial_cash=1000000,
        transaction_ratio=1,
        commission_ratio=0.0003,
        slippage_ratio=0.001,
        price_type=1)

    securities = tm.get_stock_list()
    lists=""
    for stock in securities:
        #由于掘金的回测处理问题，固定采用日线方式产生on_bar事件，
        # 在事件处理函数中进行较小周期的K线数据处理
        lists+=','+stock+'.bar.14400'

    ret = myStrategy.subscribe(lists[1:])

    #  定时处理函数无效，考虑用操作系统提供的函数实现
    # gm_set_timer_callback(on_timer)
    #gm_set_timer(1000)
    tm.backtest_model()
    ret = myStrategy.run()
    print('exit code: ', get_strerror(ret))
