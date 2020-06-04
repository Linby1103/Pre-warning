import xlrd
import os
import numpy as np


class ExcelReade(object):
    def __init__(self, excel_name, sheet_name):
        """
        # 我把excel放在工程包的当前文件夹中：
        # 1.需要先获取到工程文件的地址
        # 2.再找到excel的文件地址（比写死的绝对路径灵活）

            os.path.relpath(__file__)
            1.根据系统获取绝对路径
            2.会根据电脑系统自动匹配路径：mac路径用/,windows路径用\
            3.直接使用__file__方法是不会自动适配环境的
        """

        self.excel_path = excel_name
        # 打开指定的excel文件
        self.date = xlrd.open_workbook(self.excel_path)
        # 找到指定的sheet页
        self.table = self.date.sheet_by_name(sheet_name)
        self.rows = self.table.nrows  # 获取总行数
        self.cols = self.table.ncols  # 获取总列数

    def data_dict(self):
        if self.rows <= 1:
            print("总行数小于等于1，路径：", end='')
            print(self.excel_path)
            return False
        else:
            # 将列表的第一行设置为字典的key值
            keys = self.table.row_values(0)
            # 定义一个数组
            data = []
            # 从第二行开始读取数据，循环rows（总行数）-1次
            for i in range(1, self.rows):
                # 循环内定义一个字典，每次循环都会清空
                dict = {}
                # 从第一列开始读取数据，循环cols（总列数）次
                for j in range(0, self.cols):
                    # 将value值关联同一列的key值
                    dict[keys[j]] = self.table.row_values(i)[j] if self.table.row_values(i)[j] != "NULL" else "0"
                # 将关联后的字典放到数组里
                data.append(dict)


            return data
#Stkcd	净利润2017	净利润2018	净资产收益率	总资产净利润率	x1	x2	x3	x4	x5	x6	x7	x8	x9	x10	x11	x12	x13	x14

def GetDatafromDict():
    start = ExcelReade('./data/dataset.xlsx', 'Sheet1')
    data = start.data_dict()

    train=[]
    label=[]

    for i in range(len(data)):
        array=[]
        array.append(float(data[i]['净利润2017']))
        array.append(float(data[i]['净利润2018']))
        array.append(float(data[i]['净资产收益率']))
        array.append(float(data[i]['总资产净利润率']))
        array.append(float(data[i]['x1']))
        array.append(float(data[i]['x2']))
        array.append(float(data[i]['x3']))
        array.append(float(data[i]['x4']))
        array.append(float(data[i]['x5']))
        array.append(float(data[i]['x6']))
        array.append(float(data[i]['x7']))
        array.append(float(data[i]['x8']))
        array.append(float(data[i]['x9']))
        array.append(float(data[i]['x10']))
        array.append(float(data[i]['x11']))
        array.append(float(data[i]['x12']))
        array.append(float(data[i]['x13']))
        array.append(float(data[i]['x14']))
        train.append(array)
        if float(data[i]['净利润2017'])*float(data[i]['净利润2018'])<0:
            label.append(0)
        else:
            label.append(1)
    return np.array(train) ,np.array(label)



if __name__ == '__main__':
    start = ExcelReade('C:/Users/91324/Desktop/datasset.xlsx','Sheet1')
    data = start.data_dict()
    GetDatafromDict(data)


