import pandas as pd

data = pd.read_excel(r'C:\Users\14622\PycharmProjects\Competition\InputData.xlsx',
                     sheetname=['Pucks', 'Tickets', 'Gates'])

new_tickets = data['Tickets'].\
    merge(data['Pucks'].loc[:, ['飞机转场记录号', '到达日期', '到达类型', '到达航班']], on=['到达航班', '到达日期'], how='left').\
    merge(data['Pucks'].loc[:, ['飞机转场记录号', '出发日期', '出发类型', '出发航班']], on=['出发航班', '出发日期'], how='left')


new_tickets = new_tickets[new_tickets['飞机转场记录号_x'].notnull() & new_tickets['飞机转场记录号_y'].notnull()].\
                  loc[:, ['旅客记录号', '乘客数', '飞机转场记录号_x', '到达日期', '到达类型', '飞机转场记录号_y', '出发日期', '出发类型']]
# new_tickets.to_excel(r'C:\Users\14622\PycharmProjects\Competition\Passenger.xlsx', sheet_name='NewTickets')
# print(new_tickets)
#
#
trans = new_tickets.groupby(['飞机转场记录号_x', '飞机转场记录号_y'], as_index=False)['乘客数'].sum().\
    merge(new_tickets.loc[:, ['飞机转场记录号_x', '飞机转场记录号_y', '到达类型', '出发类型', '到达日期', '出发日期']], on=['飞机转场记录号_x', '飞机转场记录号_y'], how='left').\
    drop_duplicates(['飞机转场记录号_x', '飞机转场记录号_y'], keep='first').sort_values(['乘客数'], ascending=False).reset_index(drop=True)
# print(trans)
trans.to_excel(r'C:\Users\14622\PycharmProjects\Competition\Transfer.xlsx')
#
arrive_pucks = data['Pucks'].merge(trans.loc[:, ['飞机转场记录号_x', '乘客数']], left_on='飞机转场记录号', right_on='飞机转场记录号_x', how='left').groupby('飞机转场记录号', as_index=False)['乘客数'].sum().fillna(0)
leave_pucks = data['Pucks'].merge(trans.loc[:, ['飞机转场记录号_y', '乘客数']], left_on='飞机转场记录号', right_on='飞机转场记录号_y', how='left').groupby('飞机转场记录号', as_index=False)['乘客数'].sum().fillna(0)
new_pucks = data['Pucks'].merge(arrive_pucks, on='飞机转场记录号', how='left').merge(leave_pucks, on='飞机转场记录号', how='left')
# print(new_pucks)

new_pucks.to_excel(r'C:\Users\14622\PycharmProjects\Competition\Pucks.xlsx', index=False)

