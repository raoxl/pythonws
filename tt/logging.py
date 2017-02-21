#! /usr/bin/env python
# coding=utf-8

# ============================
# Describe :    给平台提供的日志
# D&P Author By:
# Create Date:     2016/08/01
# Modify Date:     2016/08/01
# ============================

import time
import os
import logging

print("Start test ....")
s_tm = time.time()
test_time = 10.0  # 测试时间10秒
e_tm = s_tm + 10
j = 0

pid = str(os.getpid())
while 1:
    now_time = time.time()
    j += 1
    if now_time > e_tm:
        break
        # 生成文件夹
    lujing = "e:\\test_log"
    if not os.path.exists(lujing):
        os.mkdir(lujing)

    fm2 = '%Y%m%d'
    YMD = time.strftime(fm2, time.localtime(now_time))

    filename = 'recharge_' + YMD + '.log'
    log_file = os.path.join(lujing, filename)
    t = "\t"
    log_msg = str(j) + t + str(now_time) + t + pid

    the_logger = logging.getLogger('recharge_log')
    f_handler = logging.FileHandler(log_file)
    the_logger.addHandler(f_handler)
    the_logger.setLevel(logging.INFO)
    # To pass exception information, use the keyword argument exc_info with a true value
    the_logger.info(log_msg, exc_info=False)
    the_logger.removeHandler(f_handler)

rps = j / test_time
print(rps, "rows per second")