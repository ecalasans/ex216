import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def speedup(t_s, t_p):
    return t_s/t_p

def eficiency(t_s, t_p, p):
    return t_s/(p * t_p)


n = np.array([10,20,40,80,160,320])
p = np.array([1,2,4,8,16,32,64,128])

t_serial = [i**2 for i in n]

t_parallel = []
spdup = []
ef = []

for t_s in t_serial:
    temp = []
    s_temp = []
    e_temp = []
    for j in p:
            tp = (t_s/j) + np.log2(j)
            temp.append(tp)
            s_temp.append(speedup(t_s, float(tp)))
            e_temp.append(eficiency(t_s, tp, p))
    t_parallel.append(temp)
    spdup.append(s_temp)
    ef.append(e_temp)

spdup_df = pd.DataFrame(spdup, index=n, columns=p)
ef_df = pd.DataFrame(ef, index=n, columns=p)

linhas = spdup_df.index.values.tolist()
colunas = list(spdup_df)
cores = ['']

#Speedup e Eficiência para n fixo e variando p
fig, ax = plt.subplots()

ax.set_xscale('log', basex=2)

ax.plot(colunas, spdup_df.loc[10, :] , 'r')
plt.savefig("speedup_%i_p.png" % 10)
plt.show()




