import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("data/bcc.csv")

#1. 독립표본 T-검정 양측검정인 경우 
# 사용자 코딩
group1 = df[df['Classification']==1]['Resistin']
group2 = df[df['Classification']==2]['Resistin']

group1_log = np.log(group1)
group2_log = np.log(group2)

var1 = np.var(group1_log,ddof=1)
var2 = np.var(group2_log,ddof=1)

df1 = len(group1_log)-1
df2 = len(group2_log)-1

F= var1/var2 if var1>var2 else var2/var1

print("F검정통계량:", round(F,3))

pooled_var = ((df1*var1)+(df2*var2))/(df1+df2)
print("합동 분산 추정량:", round(pooled_var,3))

mean1 = np.mean(group1_log)
mean2 = np.mean(group2_log)

n1 = len(group1_log)
n2 = len(group2_log)

df = df1+df2

t_stat = (mean1-mean2)/np.sqrt(pooled_var*(df1+df2))
p_val = 2*(1-stats.t.cdf(abs(t_stat),df))

print("t검정통계량:", round(t_stat,3))
print("p-value값:", round(p_val,3))

#2. 독립표본 t검정통계량과 p-value값 구하는 다른방법
t_stat, p_val = stats.ttest_ind(group1,group2)
print("t_stat:", round(t_stat,3))
print("p_val:", round(p_val,3))

#group1<group2
t_stat, p_val = stats.ttest_ind(group1,group2,alternative='less')
print("t_stat:", round(t_stat,3))
print("p_val:", round(p_val,3))
#group1>group2
t_stat, p_val = stats.ttest_ind(group1,group2,alternative='greater')
print("t_stat:", round(t_stat,3))
print("p_val:", round(p_val,3))

#3. 대응표본 t검정통계량과 pvalue값을 구하시오. 함수 사용법
#M1 > M2
#t_stat, p_val = stats.ttest_rel(after, before, alternative ='greater')
#M1 < M2 
#t_stat, p_val = stats.ttest_rel(after, before, alternative ='less')
#M1 = M2 
#t_stat, p_val = stats.ttest_rel(after, before)
