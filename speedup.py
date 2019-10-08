import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Funções auxiliares
def speedup(t_s, t_p):
    return t_s/t_p

def eficiency(t_s, t_p, p):
    return t_s/(p * t_p)

#Conjuntos de dados(tamanho do problema)
n = np.array([10,20,40,80,160,320])

#Número de processos/threads
p = np.array([1,2,4,8,16,32,64,128])

t_serial = [i**2 for i in n]

spdup = []
ef = []

#Cálculo do Speedup e Eficiência para os valores de n, p e t_serial
for t_s in t_serial:
    s_temp = []
    e_temp = []
    for j in p:
            #Cálculo de T_parallel
            tp = (t_s/j) + np.log2(j)

            s_temp.append(speedup(t_s, float(tp)))
            e_temp.append(eficiency(t_s, float(tp), j))
    spdup.append(s_temp)
    ef.append(e_temp)

#Construção do gráfico
spdup_df = pd.DataFrame(spdup, index=n, columns=p)
ef_df = pd.DataFrame(ef, index=n, columns=p)

linhas = spdup_df.index.values.tolist()
colunas = list(spdup_df)
cores = dict(zip(linhas,['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                         'tab:purple', 'tab:pink']))

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

ax1.set_xscale('log', basex=2)
ax2.set_xscale('log', basex=2)

for i in linhas:
    ax1.plot(colunas, spdup_df.loc[i, :], cores[i], label="n = {0}".format(i))
    ax2.plot(colunas, ef_df.loc[i, : ], cores[i], label="n = {0}".format(i))

ax1.legend()
ax2.legend()
ax2.set_xlabel("p")
ax1.set_ylabel("Speedup")
ax2.set_ylabel("Eficiência")

plt.savefig("speedup_eficiencia.png")
plt.show()




