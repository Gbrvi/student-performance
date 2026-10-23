# %%
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
import numpy as np

# %%
df = pd.read_csv('dataset_desempenho_estudantes.csv')
# INSPEÇÃO DE DADOS 
df.info()
df.describe()
df.shape

df.isna().sum()


df = df.dropna(subset="Final_Score").copy()
# %%
# -=--=--=- MEDIA, MODA e MEDIANA da NOTA FINAL -=-=-=-=--
media = df['Final_Score'].mean()
mediana = df['Final_Score'].median()
moda = df['Final_Score'].mode()

media, mediana, moda
# %%
# Dados estatistico de Final_Score
df['Final_Score'].describe()

# %% 
# HISTOGRAMA DE NOTAS

df['Final_Score'].hist(
    bins=10,
    color='skyblue',
    edgecolor='black',
    grid=False
)


plt.title("Distribuição Final Score")
plt.xlabel("Notas")
plt.ylabel("Qtd alunos")

plt.show()

# IDENTIFICANDO OUTLIERS

# %% 

q1 = df['Final_Score'].quantile(0.25)
q3 = df['Final_Score'].quantile(0.75)

IQR = q3 - q1
lim_inf = q1 - 1.5 * IQR
lim_sup = q3 + 1.5 * IQR

outliers = df[(df['Final_Score'] < lim_inf) | (df['Final_Score'] > lim_sup) ]

outliers

# %%

# PROPORÇÃO DE ESTUDANTES COM NOTA >= 70

alunos_nota_7_porcentagem = (df['Final_Score'] >= 70).mean() * 100

alunos_nota_7_porcentagem

# QUANTOS TIRARAM >= 70 DADO QUIE TEM FREQUENCIA >= 80
alunos_freq_8 = df[df['Attendance_Pct'] >= 80]
chance_passar_freq_boa = (alunos_freq_8['Final_Score'] >= 70).mean() * 100

chance_passar_freq_boa

# QUANTOS TIRARAM >= 70 DADO QUE TEM FREQUENCIA <= 80

# %%

alunos_freq_baixa = (df[df['Attendance_Pct'] <= 80])

chance_passar_freq_baixa = (alunos_freq_baixa['Final_Score'] >= 70).mean() * 100
chance_passar_freq_baixa

# CALCULANDO Z-SCORE %%

nota = 90

z = (
    nota - df['Final_Score'].mean()) / df['Final_Score'].std()

z
# %%
# AMOSTRAGEM DE DADOS 

def calcular_IC(tamanho, dados):
    amostra = dados.sample(n=tamanho)

    notas_finais = amostra['Final_Score'].dropna()
    n = len(notas_finais)

    media_amosta = notas_finais.mean()
    erro_padrao = stats.sem(notas_finais)

    intervalo = stats.t.interval(
        confidence= 0.95,
        df=n-1,
        loc=media_amosta,
        scale=erro_padrao
    )

    return {
        'n_solicitado' : tamanho,
        'n_validado': n,
        'media': media_amosta,
        'limite_inferior': intervalo[0],
        'limite_inferior': intervalo[1],
        'largura': intervalo[1] - intervalo[0]
    }

dados = df.dropna(subset=['Final_Score'])


resultado = pd.DataFrame(calcular_IC(n, dados)
                         for n in [30, 60, 120])

resultado

# %%
# HIPOTESE NULA 
# A MEDIA DAS NOTAS É == 7?

notas = df['Final_Score']

hip_nula = stats.ttest_1samp(
    notas,
    popmean=70,
    alternative="two-sided"

)
# P-valor: A probabilidade de encontrar um valor tão extremo
#          Se for, p-valor <= 0.05 isso diz que a prob de
#           Achar tal valor é muito baixo,
#           Podemos rejeitar a hipotese

# Qual é a probabilidade de eu estar observando uma media de 7 dado que minha amostra tem 88.42 de média

print("Média amostral:", notas.mean())
print("Tamanho da amostra:", len(notas))
print("Estatística t:", hip_nula.statistic)
print("P-valor:", hip_nula.pvalue)

# %%

unicos= df['Study_Method'].unique()

unicos
# Compare as notas de quem estuda individualmente com as de quem estuda em grupo.
# Use um teste t para amostras independentes de welch e examine as diferenças de médias.
# H0 a média das notas dos dois grupos é igual 


#%%
notas_individual = df[df['Study_Method'] == 'Individual']['Final_Score'].dropna()

notas_grupo = df[df['Study_Method'] == 'Study group']['Final_Score'].dropna()

notas_grupo
# %%

t_stat, p_valor = stats.ttest_ind(
    notas_individual,
    notas_grupo
)

# A estatística t mede a diferença entre as médias em relação à incerteza dessa diferença
# Proximo de zero: diferença pequena 
# T positivo ou negativo: diferença é grande em relação a incerteza
print(f"Estatística t: {t_stat:.4f}")
print(f"p-valor: {p_valor:.4f}")

# %%
# Para o exercício 14, calcule também a diferença entre médias e um intervalo de confiança 
# para essa diferença.  O resultado é estatisticamente significativo? 
# A diferença parece relevante na prática? 

#Note que: H0 a média das notas dos dois grupos é igual 


m1 = notas_individual.mean()
m2 = notas_grupo.mean()

n1 = len(notas_individual)
n2 = len(notas_grupo)

s1 = notas_individual.std()
s2 = notas_grupo.std()

diff_medias = m1 - m2


teste_t = stats.ttest_ind(notas_individual, notas_grupo, equal_var=False)
ic_95 = teste_t.confidence_interval(confidence_level=0.95)

s_pooled = np.sqrt((((n1 - 1)*s1**2) + ((n2-1)*s2**2)) / (n1+n2 - 2))

cohen_d = diff_medias / s_pooled

# Exibição dos resultados
print(f"Média Individual: {m1:.2f} | Média em Grupo: {m2:.2f}") 
print(f"Diferença entre as médias (Indiv - Grupo): {diff_medias:.2f} pontos") # 0.35 pontos de diferença apenas
print(f"IC 95% para a diferença: [{ic_95.low:.2f}, {ic_95.high:.2f}]") # Inclui zero, ou seja, pode nao ter diferença entre as medias
print(f"p-valor: {teste_t.pvalue:.4f}") # P valor alto, nao rejeitamos H0. Não temos evidencias de uma diferença entre medias.
print(f"Tamanho do efeito (d de Cohen): {cohen_d:.2f}")

# O resultado é estatisticamente significativo?
# Não. O P valor é 0.8134 é superior a 0.05 e o intervalo de confiança de 95% é de [-2.59; 3.28], isso inclui o valor zero.
# Isso mostra que nao há evidencia estatistica de diferença entre os dois métodos. 

# A diferença parece relevante na prática?
# a diferença amostral de 0,35 é irrelevante, vimos com d-cohen
    

# %%
# 16. Associação entre variáveis categóricas
# Crie uma tabela cruzada entre método de estudo e aprovação (nota ≥ 70). 
# Use o teste qui-quadrado de independência e verifique suas condições de aplicação.

df['aprovacao'] = df['Final_Score'].apply(lambda x: 'Aprovado' if x >= 70 else 'Reprovado')

tabela_cruzada = pd.crosstab(df['Study_Method'], df['aprovacao'])
print(tabela_cruzada)

pd.crosstab(df['Study_Method'], df['aprovacao'], margins=True)

# AINDA FALTA TERMINAR 
# %%
