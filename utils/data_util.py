import datetime


def get_datetime():
    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化时间字符串
    time_str = now.strftime('%m%d_%H%M')

    return time_str