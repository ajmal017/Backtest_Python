'''
learn from Udemy backtest sample
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style
import time
from datetime import timedelta
import os
from pandas.core.common import flatten
from functools import partial,reduce
import talib
# from _SB2 import *

account_size=10000
slippage=0.6 #IB Forex commision -0.00002 * position
size=1
ATR_SL=2
rr=1
perloss=account_size*0.01
plot_trades = True
plt.style.use('ggplot')
d=1


path='/Users/davidliao/Documents/code/Github/Backtest_Python/data/test'
# path='/Users/davidliao/Documents/code/Github/Backtest_Python/data/FX_data0'
# path='/Users/davidliao/Documents/code/Github/Backtest_Python/data/CMDTY_data1'


pairs_list=[]
for filename in os.listdir(path):
    if filename.endswith('.csv'):
        pairs_list.append(filename.split('.')[0])

df={}
for pair in range(len(pairs_list)):
    df[pair]=pd.read_csv(path+'/'+pairs_list[pair]+'.csv',parse_dates=['DateTime'])
    df[pair]=pd.DataFrame(df[pair])
    df[pair]['DateTime'] = pd.to_datetime(df[pair]['DateTime'],unit='s') 
    df[pair]=df[pair].set_index('DateTime')

    # plt.figure(figsize=(24,8))
    # plt.plot(df[pair]['Close'],color='black')
    # plt.xlabel(pairs_list[pair],fontsize=18)
    # plt.ylabel('Price',fontsize=18)
    



def SMA(df,fast,slow):
    df['sma_fast']=df['Close'].rolling(fast).mean()
    df['sma_slow']=df['Close'].rolling(slow).mean()
    return df

def ATR(df,n):
    df=df.copy()
    df['High-Low']=abs(df['High']-df['Low'])
    df['High-prevClose']=abs(df['High']-df['Close'].shift(1))
    df['Low-prevClose']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['High-Low','High-prevClose','Low-prevClose']].max(axis=1,skipna=False)
    df['ATR']=df['TR'].rolling(n).mean()
    df=df.drop(['High-Low','High-prevClose','Low-prevClose'],axis=1)
    return df

def RSI1(df,n):
    df['diff']=df['Close'].diff(1).dropna()
    df['gains']=np.where(df['diff']>0,df['diff'],np.nan)
    df['losses']=np.where(df['diff']<=0,df['diff'],np.nan)
    df['average_gains']=df['gains'].rolling(n,min_periods=1).mean()
    df['average_losses']=df['losses'].rolling(n,min_periods=1).mean()
    rs=abs(df['average_gains']/df['average_losses'])
    df['RSI']=100-(100/(1+rs))
    # df=df.drop(['diff','gains','losses','average_gains','average_losses'],axis=1)
    return df


def RSI2(df,n): 
    df['RSI']=talib.RSI(df['Close'], timeperiod=n)
    return df

def RSI3(df, period=3):
    # 整理資料
    Chg = df['Close']-df['Close'].shift(1)
    # Chg = Close - Close.shift(1)
    Chg_pos = pd.Series(index=Chg.index, data=Chg[Chg>0])
    Chg_pos = Chg_pos.fillna(0)
    Chg_neg = pd.Series(index=Chg.index, data=-Chg[Chg<0])
    Chg_neg = Chg_neg.fillna(0)
    
    # 計算period日平均漲跌幅度
    up_mean = []
    down_mean = []
    for i in range(period+1, len(Chg_pos)+1):
        up_mean.append(np.mean(Chg_pos.values[i-period:i]))
        down_mean.append(np.mean(Chg_neg.values[i-period:i]))
    
    # 計算 RSI
    rsi = []
    for i in range(len(up_mean)):
        rsi.append(100 * up_mean[i] / (up_mean[i]+down_mean[i]))
    rsi_series = pd.Series(index = df['Close'].index[period:], data = rsi)
    df['RSI']=rsi_series
    return df

def RSI4(df,period=3):
    #計算和上一個交易日收盤價的差值
    df['diff'] = df["Close"]-df["Close"].shift(1) 
    df['diff'].fillna(0, inplace = True)    
    df['up'] = df['diff']
    #過濾掉小於0的值
    df['up'][df['up']] = 0
    df['down'] = df['diff']
    #過濾掉大於0的值
    df['down'][df['down']] = 0
    #通過for迴圈，依次計算periodList中不同週期的RSI等值
    for period in periodList:
        df['upAvg'+str(period)] = df['up'].rolling(period).sum()/period
        df['upAvg'+str(period)].fillna(0, inplace = True)
        df['downAvg'+str(period)] = abs(df['down'].rolling(period).sum()/period)
        df['downAvg'+str(period)].fillna(0, inplace = True)
        df['RSI'] = 100 - 100/((df['upAvg'+str(period)]/df['downAvg'+str(period)]+1))
    return df

def RSI5(DF,n=3):
    "function to calculate RSI"
    df = DF.copy()
    df['delta']=df['Close'] - df['Close'].shift(1)
    df['gain']=np.where(df['delta']>=0,df['delta'],0)
    df['loss']=np.where(df['delta']<0,abs(df['delta']),0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i < n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        elif i == n:
            avg_gain.append(df['gain'].rolling(n).mean()[n])
            avg_loss.append(df['loss'].rolling(n).mean()[n])
        elif i > n:
            avg_gain.append(((n-1)*avg_gain[i-1] + gain[i])/n)
            avg_loss.append(((n-1)*avg_loss[i-1] + loss[i])/n)
    df['avg_gain']=np.array(avg_gain)
    df['avg_loss']=np.array(avg_loss)
    df['RS'] = df['avg_gain']/df['avg_loss']
    df['RSI'] = 100 - (100/(1+df['RS']))
    return df

# def isNaN(df):
#         return df['ATR'] !=df['ATR'] 

def Size(df):
    df=df.copy()
    df['R']=round(ATR_SL*df['ATR'],6)
    # df['R']=0.0006 if isNaN(df['ATR']) else round(ATR_SL*df['ATR'],6)
    df['size']=round(perloss/df['R'],0)
    return df

def trade_plot1(df, trade, exit_price, exit_date, vah, val):
    plt.figure(figsize = (25, 8))
    plt.title(trade['signal'] + ' - With result: ' + str(round(trade['result'], 2)))
    plt.plot(df['Close'][(trade['date_of_trade'] - datetime.timedelta(days = 60)): (trade['date_of_trade'] + datetime.timedelta(days = 30))], color = 'blue')
    plt.axhline(trade['TP'], color = 'green', ls = ':')
    plt.axhline(trade['SL'], color = 'red', ls = ':')
    plt.axhline(vah, color = 'black', ls = '--')
    plt.axhline(val, color = 'black', ls = '--')
    plt.axvline(df.index[np.where(df.index.time == datetime.time(0, 0))[0][-1]], color = 'black', ls = '--')
    plt.scatter(trade['date_of_trade'], trade['entry_price'], color = 'yellow', s = 200)
    plt.scatter(exit_date, exit_price, color = 'orange', s = 200)
    plt.show()
    return

def trade_plot(df, trade, exit_price, exit_date):
    plt.figure(figsize = (25, 8))
    plt.title(trade['signal'] + ' - With result: ' + str(round(trade['result'], 2)))
    plt.plot(df['Close'][(trade['date_of_trade'] - timedelta(days = 100)): (trade['date_of_trade'] + timedelta(days = 30))], color = 'blue')
    # plt.plot(df['Close'][(trade['date_of_trade'] - datetime.timedelta(days = 100)): (trade['date_of_trade'] + datetime.timedelta(days = 30))], color = 'blue')
    plt.axhline(trade['TP'], color = 'green', ls = ':')
    plt.axhline(trade['SL'], color = 'red', ls = ':')
    plt.scatter(trade['date_of_trade'], trade['entry_price'], color = 'yellow', s = 200)
    plt.scatter(exit_date, exit_price, color = 'orange', s = 200)
    
    plt.show()
    return 


for pair in range(len(pairs_list)):
    df[pair]['ATR']=ATR(df[pair],14)['ATR']
    df[pair]['sma_fast']=SMA(df[pair],30,100)
    # df[pair]['RSI1']=RSI1(df[pair],3)['RSI']
    # df[pair]['RSI']=RSI2(df[pair],3)['RSI']
    # df[pair]['RSI3']=RSI3(df[pair],3)['RSI']
    # df[pair]['RSI4']=RSI4(df[pair],3)['RSI']
    df[pair]['RSI']=RSI5(df[pair],3)['RSI']
  
    # df[pair]['size']=Size(df[pair])['size']
    # if 'XAU' not in pairs_list[pair]:
    if 'JPY' not in pairs_list[pair]:
        df[pair]['spread']=float(slippage)/float(10000)
        df[pair]['size']=float(size)*float(10000)
        print('Pair: ',pairs_list[pair],'pip:0.0001')
    else:
        df[pair]['spread']=float(slippage)/float(100)
        df[pair]['size']=float(size)*float(100)
        print('Pair: ',pairs_list[pair],'pip:0.01')
    # print(df[pair])


path1='/Users/davidliao/Documents/code/Github/Backtest_Python/report'

for pair in range(len(pairs_list)):
    df[pair].to_csv(path1+'/'+pairs_list[pair]+'_report.csv',index=1 ,float_format='%.5f')  
    # plt.figure(figsize=(24,8))
    # plt.plot(df[pair]['Close'],color='black')
    # plt.xlabel(pairs_list[pair],fontsize=18)
    # plt.ylabel('Price',fontsize=18)
    # plt.show()


csv={}
df1={}

open_trade={}
trade={}
lep={}
sep={}
ltp={}
stp={}
lsl={}
ssl={}

for pair in range(len(pairs_list)):
    csv[pair]={}
    
    open_trade[pair]=[]
    trade[pair]={}
    lep[pair]=[]
    sep[pair]=[]
    ltp[pair]=[]
    stp[pair]=[]
    lsl[pair]=[]
    ssl[pair]=[]
    for i in range(14,len(df[pair])):
        df1[pair]=df[pair][0:i] #the df1 to call strategies' functions
        # print('df1[pair]:',df1[pair])
        
        # signal,qty,entryprice,tp,sl=SB(df1[pair],d) # Call _SB2.py

        # Buy
        # if signal='BUY' and len(open_trade[pair])==0:
        if df[pair]['RSI'][i-1]>20 and df[pair]['RSI'][i]<=20 and len(open_trade[pair])==0:
        # if df[pair]['RSI'][i-1]<20 and df[pair]['RSI'][i]>=20 and len(open_trade[pair])==0:
        # if df[pair]['RSI'][i-1]<20 and df[pair]['RSI'][i]>=20 and df[pair]['sma_fast'][i-1]<df[pair]['sma_fast'][i] and len(open_trade[pair])==0:
        # if df[pair]['sma_fast'][i-1]<df[pair]['sma_slow'][i-1] and df[pair]['sma_fast'][i]>=df[pair]['sma_slow'][i] and len(open_trade[pair])==0:
            print(i,'New Long trade at price:',round(df[pair]['Close'][i],4),'On day:',df[pair].index[i],'Pair:',pairs_list[pair],'Position:',df[pair]['size'][i])
            # csv[pair][i]={'ID':i,'New Long trade at price':round(df[pair]['Close'][i],4),'On day':df[pair].index[i],'Pair':pairs_list[pair],'Position':df[pair]['size'][i]}
            trade[pair][i]={'ID':i,
                    'date_of_trade':df[pair].index[i],
                    'entry_price':round(df[pair]['Close'][i],4),
                    'signal':'Buy',
                    'result':0,
                    'TP':round(df[pair]['Close'][i]+df[pair]['ATR'][i]*ATR_SL*rr,4),
                    'SL':round(df[pair]['Close'][i]-df[pair]['ATR'][i]*ATR_SL,4)
                    }
            open_trade[pair].append(i)
            lep[pair].append(trade[pair][i]['entry_price'])
            ltp[pair].append(trade[pair][i]['TP'])
            lsl[pair].append(trade[pair][i]['SL'])

        # Sell
        if df[pair]['RSI'][i-1]<80 and df[pair]['RSI'][i]>=80 and len(open_trade[pair])==0:
        # if df[pair]['RSI'][i-1]>80 and df[pair]['RSI'][i]<=80 and len(open_trade[pair])==0:
        # if df[pair]['RSI'][i-1]>80 and df[pair]['RSI'][i]<=80 and df[pair]['sma_fast'][i-1]>df[pair]['sma_fast'][i] and len(open_trade[pair])==0:
        # if df[pair]['sma_fast'][i-1]>df[pair]['sma_slow'][i-1] and df[pair]['sma_fast'][i]<=df[pair]['sma_slow'][i] and len(open_trade[pair])==0:
            print(i,'New Short trade at price:',round(df[pair]['Close'][i],4),'On day:',df[pair].index[i],'Pair:',pairs_list[pair],'Position:',df[pair]['size'][i])
            trade[pair][i]={'ID':i,
                    'date_of_trade':df[pair].index[i],
                    'entry_price':round(df[pair]['Close'][i],4),
                    'signal':'Sell',
                    'result':0,
                    'TP':round(df[pair]['Close'][i]-df[pair]['ATR'][i]*ATR_SL*rr,4),
                    'SL':round(df[pair]['Close'][i]+df[pair]['ATR'][i]*ATR_SL,4)
                    }
            open_trade[pair].append(i)
            sep[pair].append(trade[pair][i]['entry_price'])
            stp[pair].append(trade[pair][i]['TP'])
            ssl[pair].append(trade[pair][i]['SL'])
        
        # Exit trades ----------------------------------------------------------------------------------------------
        # Buy profit
        if any(y<=df[pair]['High'][i] for y in ltp[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Buy':
                    if df[pair]['High'][i]>=trade[pair][j]['TP']:
                        trade[pair][j].update({'result':(trade[pair][j]['TP']-trade[pair][j]['entry_price']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Long profit at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With profit:',round(trade[pair][j]['result'],4),'pips:',round((trade[pair][j]['TP']-trade[pair][j]['entry_price']-df[pair]['spread'][i])*10000,0),'Spread:',df[pair]['spread'][i]*df[pair]['size'][j],'\n')
                        open_trade[pair].remove(j)
                        ltp[pair].remove(trade[pair][j]['TP'])
                        lsl[pair].remove(trade[pair][j]['SL'])
                        # plot trade
                        # if plot_trades == True:
                        #     trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])

        # Buy loss
        if any(y>=df[pair]['Low'][i] for y in lsl[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Buy':
                    if df[pair]['Low'][i]<=trade[pair][j]['SL']:
                        trade[pair][j].update({'result':(trade[pair][j]['SL']-trade[pair][j]['entry_price']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Long loss at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With loss:',round(trade[pair][j]['result'],4),'pips:',round((trade[pair][j]['SL']-trade[pair][j]['entry_price']-df[pair]['spread'][i])*10000,0),'Spread:',df[pair]['spread'][i]*df[pair]['size'][j],'\n')
                        open_trade[pair].remove(j)
                        ltp[pair].remove(trade[pair][j]['TP'])
                        lsl[pair].remove(trade[pair][j]['SL'])
                        # plot trade
                        # if plot_trades == True:
                        #     trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])

        # Sell profit
        if any(y>=df[pair]['Low'][i] for y in stp[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Sell':
                    if df[pair]['Low'][i]<=trade[pair][j]['TP']:
                        trade[pair][j].update({'result':(trade[pair][j]['entry_price']-trade[pair][j]['TP']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Short profit at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With profit:',round(trade[pair][j]['result'],4),'pips:',round((trade[pair][j]['entry_price']-trade[pair][j]['TP']-df[pair]['spread'][i])*10000,0),'Spread:',df[pair]['spread'][i]*df[pair]['size'][j],'\n')
                        open_trade[pair].remove(j)
                        stp[pair].remove(trade[pair][j]['TP'])
                        ssl[pair].remove(trade[pair][j]['SL'])
                        # plot trade
                        # if plot_trades == True:
                        #     trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])

        # Sell loss
        if any(y<=df[pair]['High'][i] for y in ssl[pair]):
            for j in open_trade[pair]:
                if trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Sell':
                    if df[pair]['High'][i]>=trade[pair][j]['SL']:
                        trade[pair][j].update({'result':(trade[pair][j]['entry_price']-trade[pair][j]['SL']-df[pair]['spread'][i])*df[pair]['size'][i]})
                        print(j,
                            'Short loss at price:',round(df[pair]['Close'][i],4),
                            'On day:',df[pair].index[i],
                            'With loss:',round(trade[pair][j]['result'],4),'pips:',round((trade[pair][j]['entry_price']-trade[pair][j]['SL']-df[pair]['spread'][i])*10000,0),'Spread:',df[pair]['spread'][i]*df[pair]['size'][j],'\n')
                        open_trade[pair].remove(j)
                        stp[pair].remove(trade[pair][j]['TP'])
                        ssl[pair].remove(trade[pair][j]['SL'])
                        # plot trade
                        # if plot_trades == True:
                        #     trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])

        # Exit after time
        # if len(open_trade[pair]) != 0:
        #     for j in open_trade[pair]:
        #         if (i-trade[pair][j]['ID']) >= 12 and trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Buy': 
        #             trade[pair][j].update({'result':(df[pair]['Close'][i]-trade[pair][j]['entry_price']-df[pair]['spread'][i])*df[pair]['size'][i]})
        #             print(j,
        #                 'Long exit after 12 bars:',round(df[pair]['Close'][i],4),
        #                 'On day:',df[pair].index[i],
        #                 'With profit:',round(trade[pair][j]['result'],4),'\n')
        #             open_trade[pair].remove(j)
        #             ltp[pair].remove(trade[pair][j]['TP'])
        #             lsl[pair].remove(trade[pair][j]['SL'])
        #             # plot trade
        #             if plot_trades == True:
        #                 trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])
        #         elif (i-trade[pair][j]['ID']) >= 12 and trade[pair][j].get('result',{})==0 and trade[pair][j].get('signal',{})=='Sell': 
        #             trade[pair][j].update({'result':(trade[pair][j]['entry_price']-df[pair]['Close'][i]-df[pair]['spread'][i])*df[pair]['size'][i]})
        #             print(j,
        #                 'Short exit after 12 bars:',round(df[pair]['Close'][i],4),
        #                 'On day:',df[pair].index[i],
        #                 'With profit:',round(trade[pair][j]['result'],4),'\n')
        #             open_trade[pair].remove(j)
        #             stp[pair].remove(trade[pair][j]['TP'])
        #             ssl[pair].remove(trade[pair][j]['SL'])
        #             # plot trade
        #             if plot_trades == True:
        #                 trade_plot(df[pair][i-1000:i+30],trade[pair][j],df[pair]['Close'][i],df[pair].index[i])
                   

pairs_results={}
profits={}
losses={}
be={}
plt.figure(figsize=(26,10))

for pair in range(len(pairs_list)):
    profits[pair]=[]
    losses[pair]=[]
    be[pair]=[]


    pairs_results[pair]=pd.DataFrame.from_dict({(i,j):trade[pair][j] for j in trade[pair].keys()},orient='index')
    pairs_results[pair]=pairs_results[pair].drop(['signal','ID','TP','SL',],axis=1)
    pairs_results[pair].set_index('date_of_trade',inplace=True)
    pairs_results[pair]['cum_res']=pairs_results[pair]['result'].cumsum()+account_size
    # print(pairs_results[pair])
  

    for t in trade[pair]:
        profits[pair].append(trade[pair][t]['result']) if trade[pair][t]['result']>0.1 else ''
        losses[pair].append(trade[pair][t]['result']) if trade[pair][t]['result']<-0.1 else ''
        be[pair].append(trade[pair][t]['result']) if -0.1<= trade[pair][t]['result']<=0.1 else ''
    
    print('---------------------------------------------------')
    print('Pair:',pairs_list[pair])
    print('wins:',len(profits[pair]))
    print('losses:',len(losses[pair]))
    print('breakevens:',len(be[pair]))
    print('win rate:{:.2%}'.format(len(profits[pair])/(len(profits[pair])+len(losses[pair]))))
  
    print('wins average:',round(np.mean(profits[pair]),2))
    print('losses average:',-1*round(np.mean(losses[pair]),2))
    print('RR:',-1*round((np.mean(profits[pair]))/(np.mean(losses[pair])),2))
    print('MDD:',round((pairs_results[pair]['cum_res'].cummax() - pairs_results[pair]['cum_res']).max(),0))
    print('MDD%:{:.2%}'.format(round((((pairs_results[pair]['cum_res'].cummax() - pairs_results[pair]['cum_res']) / pairs_results[pair]['cum_res'].cummax()).max()),2)))
    print('pairs:',pairs_list[pair],' gains/losses:',round(pairs_results[pair]['cum_res'][-1]-account_size,0))
    print('pairs:',pairs_list[pair],' gains/losses%:{:.2%}'.format(round((pairs_results[pair]['cum_res'][-1]-account_size)/account_size,2)))

    plt.plot(pairs_results[pair]['cum_res'],label=pairs_list[pair])
    plt.legend()
    plt.title('Return of each pair',fontsize=18)

my_reduce=partial(pd.merge,on='date_of_trade',how='outer')
strategy_results=reduce(my_reduce,pairs_results.values())
strategy_results=strategy_results.sort_index()
strategy_results['final_res']=strategy_results.filter(like='result',axis=1).sum(axis=1)
strategy_results['cum_res']=strategy_results['final_res'].cumsum()+account_size

print(strategy_results)

profits_keys=list(profits.keys())
profits_values=[profits[x] for x in profits_keys]
str_profits=list(flatten(profits_values))

losses_keys=list(losses.keys())
losses_values=[losses[x] for x in losses_keys]
str_losses=list(flatten(losses_values))

be_keys=list(be.keys())
be_values=[be[x] for x in be_keys]
str_be=list(flatten(be_values))

print('Strategy returns:',round(strategy_results['cum_res'][-1])-account_size)
plt.show()

# csv_df={}
# for pair in range(len(pairs_list)):
#     # print('CSV:',csv[pair])
#     csv_df[pair] = pd.DataFrame(csv[pair])
#     print(csv_df[pair])
