# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *

# 设置token
set_token('c631be98d34115bd763033a89b4b632cef5e3bb1')
# 查询历史行情
data = history(symbol='SHSE.600000',
               frequency='tick',
               start_time='2015-01-01', end_time='2018-12-31',
               fields=None,df=True)
print(data)
