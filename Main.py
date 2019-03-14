import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import mpl
import copy
import random
import pandas as pd
from datetime import timedelta
from datetime import datetime


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

GAP_TIME = timedelta(minutes=45)


def get_gates_dict(data):
    """
    读取登机门信息
    :return: key为登机口名称， value为gate实例
    """
    gates_dict = {}
    for i in range(len(data['Gates'])):
        one_row = data['Gates'].iloc[i, :]
        gate = Gate(name=one_row['登机口'],
                    terminal=one_row['终端厅'],
                    region=one_row['区域'],
                    arrive_type=one_row['到达类型'],
                    leave_type=one_row['出发类型'],
                    plane_type=one_row['机体类别'])
        gates_dict[one_row['登机口']] = gate
    return gates_dict


def get_planes_dict(pucks):
    """
    读取每架飞机的信息
    :return:
    """

    """
    宽体机（Wide-body）：332, 333, 33E, 33H, 33L, 773
    窄体机（Narrow-body）：319, 320, 321, 323, 325, 738, 73A, 73E, 73H, 73L
    """
    planes_dict = {}

    for i in range(len(pucks)):
        one_row = pucks.iloc[i, :]
        arrive_time = str(one_row['到达时刻']).split(':')[:2]
        leave_time = str(one_row['出发时刻']).split(':')[:2]
        arrive_date_time = datetime(one_row['到达日期'].year,
                                    one_row['到达日期'].month,
                                    one_row['到达日期'].day,
                                    int(arrive_time[0]),
                                    int(arrive_time[1])
                                    )
        leave_date_time = datetime(one_row['出发日期'].year,
                                   one_row['出发日期'].month,
                                   one_row['出发日期'].day,
                                   int(leave_time[0]),
                                   int(leave_time[1])
                                   )
        if str(one_row['飞机型号']) in ['332', '333', '33E', '33H', '33L', '773']:
            plane_type = 'W'
        else:
            plane_type = 'N'
        if arrive_date_time.day == 20 or leave_date_time.day == 20 or (arrive_date_time.day == 19 and leave_date_time.day == 21):
            plane = Plane(name=one_row['飞机转场记录号'],
                          arrive_time=arrive_date_time,
                          leave_time=leave_date_time,
                          arrive_type=one_row['到达类型'],
                          leave_type=one_row['出发类型'],
                          plane_type=plane_type,
                          num_arrive_peo=one_row['乘客数_x'],
                          num_leave_peo=one_row['乘客数_y'])
            planes_dict[one_row['飞机转场记录号']] = plane
    return planes_dict


def get_passengers_dict(passengers):
    """
    读取所有需要换乘乘客的信息
    :return:
    """
    passengers_dict = {}
    for i in range(len(passengers)):
        one_row = passengers.loc[i, :]
        if one_row['到达日期'].day == 20 or one_row['出发日期'] == 20:
            passenger = Passenger(arrive_plane=one_row['飞机转场记录号_x'],
                                  leave_plane=one_row['飞机转场记录号_y'],
                                  arrive_type=one_row['到达类型'],
                                  leave_type=one_row['出发类型'],
                                  arrive_date=one_row['到达日期'],
                                  leave_date=one_row['出发日期'],
                                  num_passenger=one_row['乘客数'])
            passengers_dict[i] = passenger
    return passengers_dict


def timedelta_to_hours(time_delta):
    """
    时间差转换为小时
    :param time_delta:
    :return:
    """
    return time_delta.days * 24 + time_delta.seconds / 3600


class Gate:
    """
    登机口类
    """
    def __init__(self, name, terminal, region, arrive_type, leave_type, plane_type):
        self.name = name
        self.terminal = terminal
        self.region = region
        self.arrive_type = arrive_type
        self.leave_type = leave_type
        self.plane_type = plane_type

        if len(arrive_type) != len(leave_type):
            self.level = 2
        elif len(arrive_type) == 1 and len(arrive_type) == 1:
            self.level = 1
        else:
            self.level = 3

        self.ever_used = False
        self.free_time_sections = [[datetime(2018, 1, 19, 0, 0), datetime(2018, 1, 22, 0, 0)]]
        self.record = {}
        self.receive_planes = []

    def _check_free(self, plane):
        """
        检查登机口有没有满足飞机停留的窗口时间
        若有则返回True， 剩余空窗时间， 登机口最新空窗时间
        :param plane:
        :return:
        """
        for each_section in self.free_time_sections:
            if each_section[0] <= plane.arrive_time and each_section[1] >= plane.leave_time:
                new_free_sections = copy.deepcopy(self.free_time_sections)
                new_free_sections.remove(each_section)
                new_free_sections.append([each_section[0], plane.arrive_time])
                new_free_sections.append([plane.leave_time, each_section[1]])

                new_free_sections = sorted(new_free_sections, key=lambda x: x[0])
                return True, (plane.arrive_time - each_section[0] + each_section[1] - plane.leave_time), new_free_sections
        return False, -1, None

    def cal_use_rate(self):
        """
        计算一个登机口在20日的使用时间占比
        :return:
        """
        free_time = timedelta(minutes=0)
        for each_section in self.free_time_sections:
            if (each_section[0] < datetime(2018, 1, 20)) and (each_section[1] > datetime(2018, 1, 21)):
                free_time += datetime(2018, 1, 21) - datetime(2018, 1, 20)
            elif (each_section[0] < datetime(2018, 1, 20)) and (each_section[1] > datetime(2018, 1, 20)) and (each_section[1] < datetime(2018, 1, 21)):
                free_time += each_section[1] - datetime(2018, 1, 20)
            elif (each_section[0] > datetime(2018, 1, 20)) and (each_section[1] < datetime(2018, 1, 21)):
                free_time += each_section[1] - each_section[0]
            elif (each_section[0] > datetime(2018, 1, 20)) and (each_section[0] < datetime(2018, 1, 21)) and (each_section[1] > datetime(2018, 1, 21)):
                free_time += datetime(2018, 1, 21) - each_section[0]

        use_rate = 1 - (free_time.days * 24 + free_time.seconds / 3600) / 24
        if use_rate < 0:
            print(self.free_time_sections)
        return use_rate

    def is_allowed_in_gate(self, plane):
        """
        判断飞机能否进该登机口
        :param plane:
        :return:
        """
        if (plane.arrive_type in self.arrive_type) and (plane.leave_type in self.leave_type) and (plane.plane_type in self.plane_type):
            return self._check_free(plane)
        else:
            return False, -1, None

    def in_gate(self, plane, gate_info_dict):
        """
        飞机进站，更新登机口信息
        :param plane:
        :param gate_info_dict: {'gate': each,'gap_time': gap_time,'new_free_sections': new_free_sections}
        :return:
        """

        self.free_time_sections = gate_info_dict['new_free_sections']
        self.ever_used = True
        arrive = max(0, timedelta_to_hours(plane.arrive_time - datetime(2018, 1, 20)))
        leave = min(24, timedelta_to_hours(plane.leave_time - datetime(2018, 1, 20)))
        self.record[plane.name] = [arrive, leave - arrive]
        self.receive_planes.append(plane)


class Passenger:
    """
    乘客类
    具有相同换乘的乘客归为一类
    """

    def __init__(self, arrive_plane, leave_plane, num_passenger, arrive_type, leave_type, arrive_date, leave_date):
        self.arrive_plane = arrive_plane
        self.leave_plane = leave_plane
        self.num_passenger = num_passenger
        self.arrive_type = arrive_type
        self.leave_type = leave_type
        self.arrive_date = arrive_date
        self.leave_date = leave_date

        self.arrive_gate = None
        self.leave_gate = None

    def set_gate(self, arrive_gate, leave_gate):
        """
        记录该类乘客到达航班和出发航班的去向
        :return:
        """
        self.arrive_gate = arrive_gate
        self.leave_gate = leave_gate


class Plane:
    def __init__(self, name, arrive_time, leave_time, arrive_type, leave_type, plane_type, num_arrive_peo, num_leave_peo):
        self.name = name
        self.arrive_time = arrive_time
        self.leave_time = leave_time
        self.arrive_type = arrive_type
        self.leave_type = leave_type
        self.plane_type = plane_type
        self.num_arrive_peo = num_arrive_peo
        self.num_leave_peo = num_leave_peo

        self.gate = None
        self.status = None

    def record_plane(self, gate, status):
        """
        记录飞机到达登机口的信息
        :param gate:
        :param status:
        :return:
        """
        # print('%s\t%s\t%s\t%s\t%s' %(self.name, gate, str(self.leave_time - self.arrive_time), self.arrive_time, self.leave_time))
        self.gate = gate
        self.status = status


class Airport:
    def __init__(self, gates_dict, plane_dict, passengers_dict, planes_order):
        self.gates_dict = gates_dict
        self.plane_dict = plane_dict
        self.passergens_dict = passengers_dict
        self.planes_order = planes_order

        self.avai_gates_dict = {}
        self.plane_to_gate_dict = {}
        self.result = {}

    def _update_avai_gates_list(self, plane):
        """
        获取当前可用登机口
        :param plane:
        :return:
        """

        self.avai_gates_dict = {}
        for each in self.gates_dict:
            is_allowed, gap_time, new_free_sections = self.gates_dict[each].is_allowed_in_gate(plane)
            if is_allowed:
                self.avai_gates_dict[each] = {'gate': each,
                                              'gap_time': gap_time,
                                              'new_free_sections': new_free_sections}

    def choose_best_gate(self, plane):
        """
        从可用的登机口中获取最优登机口
        :param plane:
        :return:登机口的名称, 进入登机口时间
        """
        self._update_avai_gates_list(plane)
        # 完全随机选择
        if len(self.avai_gates_dict) == 0:
            return 'fail', 0

        return sorted(self.avai_gates_dict.items(), key=lambda x: x[1]['gap_time'])[0][1], 1

        '''
        # 根据策略筛选

        level3_keys = []
        level2_keys = []
        level1_keys = []

        # 将登机口分等级
        for each in self.avai_gates_dict:
            if self.gates_dict[each].level == 3:
                level3_keys.append(each)
            elif self.gates_dict[each].level == 2:
                level2_keys.append(each)
            else:
                level1_keys.append(each)

        # 先选择等级高 且 使用过 的的登机口
        for each in level1_keys:
            if self.gates_dict[each].ever_used:
                return each, plane.arrive_time

        for each in level2_keys:
            if self.gates_dict[each].ever_used:
                return each, plane.arrive_time

        for each in level3_keys:
            if self.gates_dict[each].ever_used:
                return each, plane.arrive_time

        # 全部没使用过， 在等级高的登机口中随机选择一个

        for each in level1_keys:
            return each, plane.arrive_time

        for each in level2_keys:
            return each, plane.arrive_time

        for each in level3_keys:
            return each, plane.arrive_time
        '''

    def run(self):
        """
        飞机进站
        :return:
        """
        # for each_plane in list(self.plane_dict.keys())[:self.test_size]:
        for each_plane in [x for j in self.planes_order for x in j]:

            plane = self.plane_dict[each_plane]
            gate_info_dict, status = self.choose_best_gate(plane)

            if status == 1:
                gate = gate_info_dict['gate']
                self.gates_dict[gate_info_dict['gate']].in_gate(plane, gate_info_dict)
                self.plane_dict[each_plane].record_plane(gate_info_dict['gate'], status)

            else:
                gate = gate_info_dict
                self.plane_dict[each_plane].record_plane(gate_info_dict, status)

            self.plane_to_gate_dict[each_plane] = gate

        for each_passenger in self.passergens_dict:
            passenger = self.passergens_dict[each_passenger]
            self.passergens_dict[each_passenger].set_gate(self.plane_to_gate_dict[passenger.arrive_plane],
                                                          self.plane_to_gate_dict[passenger.leave_plane])

    ################# 以下三个函数 计算机场内登机口使用情况、飞机分配情况、乘客换乘情况 ##################

    def cal_gates(self):
        """
        统计使用的登机口数
        :return:
        """
        count = 0
        for each in self.gates_dict:
            if self.gates_dict[each].ever_used:
                count += 1
        self.result['使用的登机口数'] = count
        self.result['未使用登机口数'] = 69 - count

    def cal_planes(self):
        """
        统计飞机准时进入登机口、等待后进入登机口、延误起飞的数目
        :return:
        """
        on_time = 0
        fail = 0

        for each_plane in list(self.plane_dict):
            plane = self.plane_dict[each_plane]
            if plane.status == 1:
                on_time += 1
            else:
                fail += 1

        self.result['分配成功'] = on_time
        self.result['分配失败'] = fail
        self.result['分配失败率'] = fail / (on_time + fail)
        self.result['分配成功率'] = on_time / (on_time + fail)

    def cal_passengers(self, matrix):
        """
        计算换乘乘客总人数、被分配到停机坪人数、换乘失败人数、紧张度等
        :param matrix: 流程时间、捷运时间、步行时间 矩阵
        :return:
        """
        num_passengers = 0
        num_trans_failed = 0
        total_tensity = 0
        num_loss = 0
        total_flow_time = 0

        transfer_time_list = []
        tensity_list = []

        for each_passengers in self.passergens_dict:
            passenger = self.passergens_dict[each_passengers]
            if passenger.arrive_gate != 'fail' and passenger.leave_gate != 'fail':
                arrive_gate = self.gates_dict[passenger.arrive_gate]
                leave_gate = self.gates_dict[passenger.leave_gate]
                arrive_plane = self.plane_dict[passenger.arrive_plane]
                leave_plane = self.plane_dict[passenger.leave_plane]
                num_passengers += passenger.num_passenger

                # 流程时间
                flow_time = matrix['Flow'].loc["%s%s" % (passenger.arrive_type, arrive_gate.terminal)]["%s%s" % (passenger.leave_type, leave_gate.terminal)]
                total_flow_time += flow_time * passenger.num_passenger
                # 捷运时间
                metro_time = 8 * matrix['Metro'].loc["%s%s" % (passenger.arrive_type, arrive_gate.terminal)]["%s%s" % (passenger.leave_type, leave_gate.terminal)]
                # 步行时间
                walk_time = matrix['Walk'].loc['%s-%s' % (arrive_gate.terminal, arrive_gate.region)]['%s-%s' % (leave_gate.terminal, leave_gate.region)]

                # 乘客总换乘时间
                total_transfer_time = (flow_time + metro_time + walk_time) / 60
                # 换乘航班连接时间
                transfer_gap_time = timedelta_to_hours(leave_plane.leave_time - arrive_plane.arrive_time)
                if total_transfer_time > transfer_gap_time:
                    num_trans_failed += passenger.num_passenger
                    # 换乘失败，换乘时间记为6小时
                    total_transfer_time = 6
                transfer_time_list.extend([total_transfer_time * 60] * passenger.num_passenger)

                tensity = total_transfer_time / transfer_gap_time
                tensity_list.extend([tensity] * passenger.num_passenger)
                total_tensity += passenger.num_passenger * (total_transfer_time / transfer_gap_time)

            else:
                num_loss += passenger.num_passenger

        self.result['总换乘人数'] = num_passengers
        self.result['停机坪人数'] = num_loss
        self.result['换乘失败人数'] = num_trans_failed
        self.result['总体紧张度'] = total_tensity
        self.result['平均紧张度'] = total_tensity / num_passengers
        self.result['总流程时间'] = total_flow_time
        self.result['平均流程时间'] = total_flow_time / num_passengers
        self.result['平均紧张度（不含停机坪）'] = total_tensity / (num_passengers - num_loss)
        self.result['平均流程时间（不含停机坪）'] = total_flow_time / (num_passengers - num_loss)
        self.result['换乘时间分布'] = transfer_time_list
        self.result['换乘紧张度分布'] = tensity_list
    #################################################################################

    ########################## 以下几个函数实现结果可视化 ############################################

    def plot_nw_bar(self):
        """
        绘制窄登机口、宽登机口分配到的航班数量
        :return:
        """
        W_dict = {}
        N_dict = {}
        for each in self.plane_dict:
            plane = self.plane_dict[each]
            if plane.plane_type == 'W':
                if plane.gate not in W_dict:
                    W_dict[plane.gate] = 1
                else:
                    W_dict[plane.gate] += 1
            elif plane.plane_type == 'N':
                if plane.gate not in N_dict:
                    N_dict[plane.gate] = 1
                else:
                    N_dict[plane.gate] += 1
        plt.figure()
        ax = plt.subplot()
        plt.bar(W_dict.keys(), W_dict.values())
        plt.annotate("宽体机成功分配到登机口的数量：%d, 比例为%.3f" % (sum(W_dict.values()) - W_dict['fail'], (1 - W_dict['fail'] / sum(W_dict.values()))), xy=(0, 6), xytext=(0, 6))
        for xy in zip(W_dict.keys(), W_dict.values()):
            plt.annotate("%d" % xy[1], xy=xy, xytext=xy, verticalalignment="bottom", horizontalalignment="center")
        plt.xlabel('登机口')
        plt.ylabel('数量')
        plt.title('宽体机分配到的登机口及数量')
        plt.savefig('./output/宽体机分配到的登机口及数量.png')

        plt.figure(figsize=(15, 7))
        ax = plt.subplot()
        plt.bar(N_dict.keys(), N_dict.values())
        plt.annotate("窄体机成功分配到登机口的数量：%d, 比例为%.3f" % (sum(N_dict.values()) - N_dict['fail'], (1 - N_dict['fail'] / sum(N_dict.values()))), xy=(0, 35), xytext=(0, 35), fontsize=15)
        for xy in zip(N_dict.keys(), N_dict.values()):
            plt.annotate("%d" % xy[1], xy=xy, xytext=xy, verticalalignment="bottom", horizontalalignment="center")
        plt.xlabel('登机口')
        plt.ylabel('数量')
        plt.title('窄体机分配到的登机口及数量')
        plt.savefig('./output/窄体机分配到的登机口及数量.png')
        # plt.show()

    def plot_ts_use_rate(self):
        """
        绘制T，S登机口的20号内的使用比例
        :return:
        """
        rate_list = []
        for gate in self.gates_dict:
            use_rate = self.gates_dict[gate].cal_use_rate()
            rate_list.append(use_rate)

        rate = pd.DataFrame({'使用率': rate_list}, index=list(self.gates_dict.keys())).sort_values(by=['使用率'], ascending=False)
        self.result['登机口平均使用率'] = sum(rate_list) / len(rate_list)
        f, ax = plt.subplots(figsize=(20, 8))
        rate.plot(kind='bar', ax=ax)
        ax.set_title('20日各登机口使用率'), ax.set_xlabel('登机口'), ax.set_ylabel('使用率')
        f.savefig('./output/20日各登机口使用率.png', dpi=400)

    def plot_gantt(self):
        """
        绘制各登机口使用甘特图
        :return:
        """
        height = 0.8
        interval = 0.2
        f, ax = plt.subplots(figsize=(20, 10))
        for i, each_gate in enumerate(list(self.gates_dict)):
            gate = self.gates_dict[each_gate]
            for index, each_plane in enumerate(list(gate.record)):
                ax.broken_barh([gate.record[each_plane]], [(height + interval) * i - 0.5, height])
                plt.text(gate.record[each_plane][0], (height + interval) * i - 0.5, each_plane)
        ax.set_title('20日各登机口使用情况'), ax.set_xlabel('时间段'), ax.set_ylabel('登机口')
        ax.set_xlim(0, 24), ax.set_xticks(range(0, 25, 1))
        ax.set_yticklabels(list(self.gates_dict.keys())), ax.set_yticks(range(0, 70, 1))
        f.savefig('./output/20日各登机口使用情况.png', dpi=400)

    def plot_distribution_transfer(self):
        """
        绘制乘客换乘时间分布图
        :return:
        """
        f, ax = plt.subplots(figsize=(8,5))
        plt.tight_layout(pad=5)
        range_ = list(range(0, 100, 5))
        group = pd.cut(self.result['换乘时间分布'], range_, right=True)
        rate = group.value_counts() / len(self.result['换乘时间分布'])
        rate.plot(kind='bar', ax=ax)
        cum_rate = rate.cumsum()
        cum_rate.plot(ax=ax)
        for xy in zip(range(20), cum_rate.values):
            plt.annotate("%.2f" % xy[1], xy=xy, xytext=xy, verticalalignment="bottom", horizontalalignment="center")

        # ax.set_xticklabels(['(0,%d]' % i for i in range(5, 100, 5)])
        ax.set_title('总体旅客换乘时间分布图'), ax.set_xlabel('换乘时间（分钟）'), ax.set_ylabel('比率')
        plt.xticks(rotation=90)
        f.savefig('./output/总体旅客换乘时间分布图.png', dpi=200)

    def plot_distribution_tensity(self):
        """
        绘制乘客换乘紧张度分布图
        :return:
        """
        f, ax = plt.subplots(figsize=(8,5))
        plt.tight_layout(pad=5)
        range_ = [x * 0.1 for x in range(9)]
        group = pd.cut(self.result['换乘紧张度分布'], range_, right=True)
        rate = group.value_counts() / len(self.result['换乘紧张度分布'])
        rate.plot(kind='bar', ax=ax)
        cum_rate = rate.cumsum()
        cum_rate.plot(ax=ax)
        for xy in zip(range(9), cum_rate.values):
            plt.annotate("%.2f" % xy[1], xy=xy, xytext=xy, verticalalignment="bottom", horizontalalignment="center")
        # ax.set_xticklabels(['(0,%.1f]' % (i * 0.1) for i in range(1, 9)])
        ax.set_title('总体旅客换乘紧张度分布图'), ax.set_xlabel('换乘紧张度'), ax.set_ylabel('比率')
        f.savefig('./output/总体旅客换乘紧张度分布图.png', dpi=200)

    def generate_result(self, data, name):
        """
        生成转场记录号对应的登机口
        :return:
        """
        df = pd.DataFrame({'飞机转场记录号': list(self.plane_to_gate_dict.keys()), '登机口': list(self.plane_to_gate_dict.values())})
        new_data = data['Pucks'].merge(df, on='飞机转场记录号', how='left')
        # print(new_data)
        new_data.loc[:, ['飞机转场记录号', '登机口']].to_excel(r'./output/问题%s登机口.xlsx' % name)


def generate_one_solution(pucks, alpha_gate, alpha_length, alpha_passengers, N=31):
    """
    根据参数，生成一个航班的优先级排序
    :return:
    """

    """
    宽体机（Wide-body）：332, 333, 33E, 33H, 33L, 773
    窄体机（Narrow-body）：319, 320, 321, 323, 325, 738, 73A, 73E, 73H, 73L
    """
    gate_tensity_dict = {
        'DIW': 4.25,
        'DIN': 27.00,
        'IDW': 6.5,
        'IDN': 29.5,
        'DDW': 0,
        'DDN': 8.46,
        'IIW': 3.32,
        'IIN': 10.67
    }
    planes_dict = {}

    for i in range(len(pucks)):
        one_row = pucks.iloc[i, :]
        arrive_time = str(one_row['到达时刻']).split(':')[:2]
        leave_time = str(one_row['出发时刻']).split(':')[:2]
        arrive_date_time = datetime(one_row['到达日期'].year,
                                    one_row['到达日期'].month,
                                    one_row['到达日期'].day,
                                    int(arrive_time[0]),
                                    int(arrive_time[1])
                                    )
        leave_date_time = datetime(one_row['出发日期'].year,
                                   one_row['出发日期'].month,
                                   one_row['出发日期'].day,
                                   int(leave_time[0]),
                                   int(leave_time[1])
                                   )
        if str(one_row['飞机型号']) in ['332', '333', '33E', '33H', '33L', '773']:
            plane_type = 'W'
        else:
            plane_type = 'N'
        if arrive_date_time.day == 20 or leave_date_time.day == 20 or (
                        arrive_date_time.day == 19 and leave_date_time.day == 21):
            # 该架飞机的优先权重系数
            weight = alpha_gate * gate_tensity_dict[str(one_row['到达类型'] + str(one_row['出发类型']) + plane_type)] + \
                     alpha_length * ((leave_date_time - arrive_date_time).days * 24 + (
                         leave_date_time - arrive_date_time).seconds / 3600) + \
                     alpha_passengers * (one_row['乘客数_x'] + one_row['乘客数_y'])
            planes_dict[one_row['飞机转场记录号']] = weight
    plane_order = [y[0] for y in sorted(planes_dict.items(), key=lambda x: x[1], reverse=True)]
    plane_order = [plane_order[i: i + N] for i in range(0, len(plane_order), N)]
    return plane_order


def evaluation(planes_order, plot=False):
    """
    计算在某一参数对下的各个优化目标的值
    :param planes_order:
    :param plot:
    :return:
    """
    data = pd.read_excel(r'InputData.xlsx',
                         sheetname=['Pucks', 'Gates'])
    pucks = pd.read_excel(r'Pucks.xlsx')
    passengers = pd.read_excel(r'Transfer.xlsx')
    matrix = pd.read_excel(r'Matrix.xlsx',
                           sheetname=['Flow', 'Metro', 'Walk'],
                           index_col='index')

    gates_dict = get_gates_dict(data)
    planes_dict = get_planes_dict(pucks)
    passengers_dict = get_passengers_dict(passengers)

    airport = Airport(gates_dict, planes_dict, passengers_dict, planes_order)
    airport.run()

    airport.cal_gates()
    airport.cal_planes()
    airport.cal_passengers(matrix)

    if plot:
        airport.plot_nw_bar()
        airport.plot_ts_use_rate()
        airport.plot_gantt()
        airport.plot_distribution_transfer()
        airport.plot_distribution_tensity()
        airport.generate_result(data, '二')
    print(airport.result)
    return airport.result


def random_search():
    """
    三个参数的随机寻优
    :return:
    """
    data = pd.read_excel(r'InputData.xlsx',
                         sheetname=['Gates'])
    pucks = pd.read_excel(r'Pucks.xlsx')
    passengers = pd.read_excel(r'Transfer.xlsx')
    matrix = pd.read_excel(r'Matrix.xlsx',
                           sheetname=['Flow', 'Metro', 'Walk'],
                           index_col='index')

    def _one_time_search(alpha_gate, alpha_length, alpha_pass):
        gates_dict = get_gates_dict(data)
        planes_dict = get_planes_dict(pucks)
        passengers_dict = get_passengers_dict(passengers)

        planes_order = generate_one_solution(pucks, alpha_gate, alpha_length, alpha_pass, 30)
        airport = Airport(gates_dict, planes_dict, passengers_dict, planes_order)
        airport.run()
        airport.cal_gates()
        airport.cal_planes()
        airport.cal_passengers(matrix)
        return airport.result


    best_alpha_gate = 0
    best_alpha_length = 0
    best_alpha_pass = 0
    best_result = None

    best_fail_rate = np.inf
    best_trans_time = np.inf
    best_used_gate = 100
    for i in range(10000):
        alpha_gate = random.uniform(50, 200)
        alpha_length = random.uniform(0, 5) * -1
        alpha_pass = random.uniform(-5, 5) * 1
        result = _one_time_search(alpha_gate, alpha_length, alpha_pass)
        if result['分配失败率'] <= best_fail_rate:
            if result['分配失败率'] < best_fail_rate:
                best_fail_rate = result['分配失败率']
                best_alpha_gate = alpha_gate
                best_alpha_length = alpha_length
                best_alpha_pass = alpha_pass
                best_result = result
                best_trans_time = result['总体紧张度']
                best_used_gate = result['未使用登机口数']
            elif result['总体紧张度'] <= best_trans_time:
                if result['总体紧张度'] < best_trans_time:
                    best_fail_rate = result['分配失败率']
                    best_alpha_gate = alpha_gate
                    best_alpha_length = alpha_length
                    best_alpha_pass = alpha_pass
                    best_result = result
                    best_trans_time = result['总体紧张度']
                    best_used_gate = result['未使用登机口数']
            elif result['未使用登机口数'] < best_used_gate:
                best_fail_rate = result['分配失败率']
                best_alpha_gate = alpha_gate
                best_alpha_length = alpha_length
                best_alpha_pass = alpha_pass
                best_result = result
                best_trans_time = result['总体紧张度']
                best_used_gate = result['未使用登机口数']

        print('---Generation %d ---' % i)
        print('Now params: alpha_gate=%.5f, alpha_length=%0.5f, alpha_pass=%0.5f' % (alpha_gate, alpha_length, alpha_pass))
        print('Now result:', result)
        print('Best params: alpha_gate=%.5f, alpha_length=%0.5f, alpha_pass=%0.5f' % (best_alpha_gate, best_alpha_length, best_alpha_pass))
        print('Best result:', best_result)


if __name__ == '__main__':

    # 单次计算
    pucks = pd.read_excel(r'Pucks.xlsx')
    evaluation(generate_one_solution(pucks, alpha_gate=150.44638, alpha_length=-4.98277, alpha_passengers=0.60535), True)
    #
    # 参数随机寻优
    # random_search()


