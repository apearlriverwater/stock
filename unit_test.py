import unittest
import gmTools as tls
import trade_model_v3 as tm

class Test_uint_test(unittest.TestCase):
    
    '''
        symbol_list	string	证券代码, 带交易所代码以确保唯一，如SHSE.600000，同时支持多只代码
        bar_type	int	bar周期，以秒为单位，比如60即1分钟bar
        begin_time	string	开始时间, 如2015-10-30 09:30:00
        end_time	string	结束时间, 如2015-10-30 15:00:00
    '''
    '''
    def test_read_kline(self):
        symbol_list	='SHSE.600000'
        bar_type=15*60
        begin_time='2015-10-30 09:30:00'
        end_time='2015-10-30 15:00:00'
        bars=tls.read_kline(symbol_list,bar_type,begin_time,end_time)
        print(bars[:3])
        print(bars[-3:])
    '''


    #def test_read_ticks(self):
    #    tick_file_path='e:/data/ticks-000001-20180201.dat'
    #    ticks=tls.read_ticks(tick_file_path)
    #    print(ticks[:3])
    #    print(ticks[-3:])
        

''' 
    def test_read_cap(self):
     
        file_path='e:/data/CAP-000554-015.dat'
        data=tls.read_cap_flow(file_path)
        print(data[:3])
        print(data[-3:])

        ma=[5,10,20]
        nLastWeek=3
        nCount=sum(ma)+nLastWeek*2 
        i=nCount
        col=['Date','Time']
        while i<len(data):
            if(tls.IsMaUp(data[i:i+nCount],ma,nLastWeek)):
                print ('main flow ma up\n',data[col][i+nCount-1:i+nCount])
            elif(tls.IsMaDown(data[i:i+nCount],ma,nLastWeek)):
                print ('main flow ma down\n',data[col][i+nCount-1:i+nCount])
            i=i+1

        i=i
'''

if __name__ == '__main__':
    unittest.main()
