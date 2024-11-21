from datetime import datetime, timedelta
import random
def calculate_new_time(base_time: str, time_offset: float) -> str:
    """
    计算给定时间加上偏移量后的新时间。
    
    :param base_time: 基础时间，格式为 "HH:MM:SS.sss"
    :param time_offset: 偏移量，秒的形式，可以是浮点数。
    :return: 新时间，格式为 "HH:MM:SS.sss"
    """
    # 将字符串形式的时间转换为 datetime 对象
    base_time_obj = datetime.strptime(base_time, "%H:%M:%S.%f")
    
    # 将偏移量转换为 timedelta 对象
    offset = timedelta(seconds=time_offset)
    
    # 计算新时间
    new_time_obj = base_time_obj + offset
    
    # 返回格式化的新时间
    return new_time_obj.strftime("%H:%M:%S.%f")[:-3]  # 保留毫秒，去掉多余的位数

# 示例用法
base_time = "9:47:57.823"  # 基础时间
randx = random.randint(0, 9) / 1000
print(randx)
time_offset = 6681.82 + randx  # 偏移量，秒
print(time_offset)
new_time = calculate_new_time(base_time, time_offset)
print(f"新时间: {new_time}")
