import pandas as pd
import psmatching.match as psm
import numpy as np
import random
import operator
 
random.seed(1)  # 生成同一个随机数；
road='/data/home/baldwinhe/uplift_project/datasets/Hillstrom'# 数据存放的地址
 
###### 函数部分 ######
# 样本数量均衡测试
def sample_test(dt_treat,dt_control):
        if len(dt_treat)>len(dt_control):
                print('---样本数均衡测试---')
                print('实验组样本数:%d,对照组样本数:%d,实验组需要剔除%d个样本' % (len(dt_treat),len(dt_control),len(dt_treat)-len(dt_control)))
                l = list(dt_treat.index.values)
                sap = random.sample(l, len(dt_treat) - len(dt_control))  # 随机抽取n个元素
                dt_treat = dt_treat.drop(sap)
                print('删除的数据索引为：%s' % str(sap))
                print('Done!')
                print('调整后的实验组样本数:%d,对照组样本数:%d' % (len(dt_treat),len(dt_control)))
                print('\n')
        else:
                pass
        return dt_treat
 
# 计算实验效应 % ATE
def effect_calculate(core,dt):
        y_1 = np.average(dt[dt.CASE == 1][core])
        y_0 = np.average(dt[dt.CASE == 0][core])
        ATE = y_1 - y_0
        print('实验组%s：%.4f,对照组%s：%.4f，因果效应：%.4f' % (core, y_1, core, y_0, ATE))
 
# 用户发送邮件的优先级排序
def send_user(dt):
        dt['recency_history'] = dt['recency'] + dt['history_segment']
        # 根据spend，筛选第二象限用户群（自然不消费，邮件消费）
        spd2=dt[ (dt['ITE_spend']>0)  & ( (dt['spend_mch_1']==0) & (dt['spend_mch_2']==0) & (dt['spend_mch_3']==0) ) ]
        spd2=spd2.sort_values(by=['ITE_spend'], ascending=False )
        # 根据spend，筛选第一象限用户群（自然消费，邮件也消费）
        spd1=dt[ (dt['ITE_spend']>0) & ( (dt['spend_mch_1']!=0) | (dt['spend_mch_2']!=0) | (dt['spend_mch_3']!=0) ) ]
        spd1=spd1.sort_values(by=['ITE_spend'], ascending=False)
        # 根据visit，筛选第二象限用户群（自然不访问，邮件访问）
        vit2=dt[ (dt['ITE_spend']==0) & (dt['ITE_visit']>0) & ( (dt['visit_mch_1']==0) & (dt['visit_mch_2']==0) & (dt['visit_mch_3']==0) ) ]
        vit2=vit2.sort_values(by=['ITE_visit'], ascending=False)
        # 根据visit，筛选第一象限用户群（自然访问，邮件也访问）
        vit1=dt[ (dt['ITE_spend']==0) & (dt['ITE_visit']>0) & ( (dt['visit_mch_1']!=0) | (dt['visit_mch_2']!=0) | (dt['visit_mch_3']!=0) ) ]
        vit1=vit1.sort_values(by=['ITE_visit'], ascending=False)
        # 剩下的用户按照recency和history_segment两个维度综合考虑
        rem1=dt[ (dt['ITE_spend']==0) & (dt['ITE_visit']==0)].sort_values(by=['recency_history'], ascending=False)
        rem2=dt[ (dt['ITE_spend']==0) & (dt['ITE_visit']<0)].sort_values(by=['recency_history'], ascending=False)
        rem3=dt[ dt['ITE_spend']<0 ].sort_values(by=['recency_history'], ascending=False)
        # 合并用户群
        subs = [spd2, spd1, vit2, vit1, rem1, rem2, rem3]
        df = pd.concat(subs).reset_index()
 
        df.drop(columns=['recency_history','index'],inplace=True)
        return df
 
 
 
 
###### 第一部分：数据预处理 ######
# 读取数据集
dt=pd.read_csv(road+'/EmailAnalytics.csv')
 
# 历史消费区间，文本型转类别型
dt['history_segment']=dt['history_segment'].apply(lambda x: int(x[0])-1)
 
# 区域，文本型转类别型
zip_mapping={'Rural':0,'Surburban':1,'Urban':2}
dt['zip_code']=dt['zip_code'].map(zip_mapping)
 
# 实验分组，文本型转类别型
seg_mapping={'No E-Mail':0,'Mens E-Mail':1,'Womens E-Mail':2}
dt['CASE']=dt['segment'].map(seg_mapping)
 
# 历史购买渠道，one-hot编码
dt=pd.get_dummies(dt,columns=['channel'])
 
# 删除没用的字段
dt.drop(columns=['history','segment'],inplace=True)
 
# 划分实验组和对照组
dt_ctl=dt[dt['CASE']==0]
dt_men=dt[dt['CASE']==1]
dt_wom=dt[dt['CASE']==2]
 
# 检查实验组和对照组样本比例是否均衡
dt_men=sample_test(dt_men,dt_ctl)
dt_wom=sample_test(dt_wom,dt_ctl)
 
dt_men=pd.concat([dt_men,dt_ctl])
dt_wom=pd.concat([dt_wom,dt_ctl])
 
dt_wom.loc[dt_wom['CASE']==2,'CASE']=1
 
# 保存数据集
dt_men.to_csv(road+'/Email_men.csv')
dt_wom.to_csv(road+'/Email_wom.csv')
 
 
 
 
###### 第二部分：实验分析 ######
 
# 数据集的地址，默认数据文件是csv格式，其他格式可能会报错
path=road+r'/Email_men.csv'
# 计算倾向性得分的模型格式，格式：Y~X1+X2+...+Xn，其中Y为treatment列，X为协变量列
model = "CASE ~ recency + history_segment + mens + womens + zip_code + newbie " \
        "+ channel_Phone + channel_Web + channel_Multichannel"
# k每个实验组样本所匹配的对照组样本的数量
k = "3"
 
# 初始化PSMatch实例
m = psm.PSMatch(path, model, k)
 
# 计算倾向得分，为接下来的匹配准备数据
dd=m.prepare_data()
 
# 根据倾向性得分做匹配，其中caliper代表是否有卡尺，replace代表是否是有放回采样
m.match(caliper = None, replace = True)
 
# 混淆变量与treatment做卡方检验，检验混淆变量和treatment是不是独立的
# m.evaluate()
 
# 获取匹配的样本子集
mdt = m.matched_data
mdt['OPTUM_LAB_ID']=mdt.index
mdt.index.rename('index',inplace=True)
 
# 获取匹配的样本编号
mch = m.matches
 
# 计算实验的因果效应（ATE）
effect_calculate('visit',mdt)
effect_calculate('conversion',mdt)
effect_calculate('spend',mdt)
 
# 为每个实验组和对照组样本匹配评价指标
mch=pd.merge(mch,mdt[['OPTUM_LAB_ID','visit','conversion','spend']],left_on='CASE_ID',right_on='OPTUM_LAB_ID',how='left').rename(
        columns={'visit':'visit_lab','conversion':'conversion_lab','spend':'spend_lab','OPTUM_LAB_ID':'LAB_ID'})
mch=pd.merge(mch,mdt[['OPTUM_LAB_ID','visit','conversion','spend']],left_on='CONTROL_MATCH_1',right_on='OPTUM_LAB_ID',how='left').rename(
        columns={'visit':'visit_mch_1','conversion':'conversion_mch_1','spend':'spend_mch_1'})
mch=pd.merge(mch,mdt[['OPTUM_LAB_ID','visit','conversion','spend']],left_on='CONTROL_MATCH_2',right_on='OPTUM_LAB_ID',how='left').rename(
        columns={'visit':'visit_mch_2','conversion':'conversion_mch_2','spend':'spend_mch_2'})
mch=pd.merge(mch,mdt[['OPTUM_LAB_ID','visit','conversion','spend']],left_on='CONTROL_MATCH_3',right_on='OPTUM_LAB_ID',how='left').rename(
        columns={'visit':'visit_mch_3','conversion':'conversion_mch_3','spend':'spend_mch_3'})
 
mch=mch[['LAB_ID','visit_lab','visit_mch_1','visit_mch_2','visit_mch_3',
         'conversion_lab','conversion_mch_1','conversion_mch_2','conversion_mch_3',
         'spend_lab','spend_mch_1','spend_mch_2','spend_mch_3']]
 
# 计算实验的个体因果效应（ITE）
mch['ITE_visit']=mch['visit_lab']-(mch['visit_mch_1']+mch['visit_mch_2']+mch['visit_mch_3'])/3
mch['ITE_conversion']=mch['conversion_lab']-(mch['conversion_mch_1']+mch['conversion_mch_2']+mch['conversion_mch_3'])/3
mch['ITE_spend']=mch['spend_lab']-(mch['spend_mch_1']+mch['spend_mch_2']+mch['spend_mch_3'])/3
 
mch=pd.merge(mch,dt[['OPTUM_LAB_ID','recency','history_segment','mens','womens','zip_code','newbie','channel']],
             left_on='LAB_ID', right_on='OPTUM_LAB_ID', how='left').drop(labels=['OPTUM_LAB_ID'],axis=1)
 
# 优先发送邮件的10000个用户
pri=send_user(mch).head(10000)
 
# 不应该发送邮件的10000个用户
last=send_user(mch).tail(10000)
 
