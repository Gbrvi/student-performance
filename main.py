# %%
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
import numpy as np
import seaborn as sns 

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

df['Aprovacao'] = np.where(df['Final_Score'] >= 70, 'Aprovado', 'Reprovado')

tabela_cruzada = pd.crosstab(df['Study_Method'], df['Aprovacao'])
tabela_cruzada

pd.crosstab(df['Study_Method'], df['Aprovacao'], margins=True)

# H0: O metodo de estudo e a aprovação é independente?  
# H1: Método de estudo e aprovação não são independentes

chi2, p_valor, gl, esperado = stats.chi2_contingency(tabela_cruzada)

df_esperado = pd.DataFrame(esperado, index=tabela_cruzada.index, columns=tabela_cruzada.columns)

df_esperado

# Total de células da tabela esperada
total_celulas = df_esperado.size

# Quantas células têm frequência esperada menor que 5
celulas_menores_5 = (df_esperado < 5).sum().sum()

# Proporção de células com frequência esperada menor que 5
proporcao = celulas_menores_5 / total_celulas

# Menor frequência esperada
menor_esperado = df_esperado.min().min()

print("Total de células:", total_celulas)
print("Células com esperado < 5:", celulas_menores_5)
print(f"Proporção abaixo de 5: {proporcao:.1%}")
print(f"Menor frequência esperada: {menor_esperado:.2f}")

# Se a condição falhar e for uma tabela 2x2 (ou ate mesmo 3x3), roda o Teste Exato de Fisher:
if proporcao < 0.8:
    oddsratio, p_fisher = stats.fisher_exact(tabela_cruzada)
    print(f"[FISHER] p-valor do Teste Exato de Fisher: {p_fisher:.4f}")

# H0: O metodo de estudo e a aprovação é independente?  
# ------- Com P-valor de 0.6650 é superior a 0.05, não rejeitamos a H0. não encontramos evidências estatísticas suficientes
#         Para afirmar que existe associação entre o método de estudo e a aprovação.

#-------- Todos metodo de estudo obteve notas semelhantes. 

# AINDA FALTA TERMINAR 
# %%

# 17. Correlação
# Calcule as correlações entre nota final e horas de estudo, frequência, sono e nota anterior. 
# Qual variável apresenta a maior correlação linear com a nota final?

col_alvo = 'Final_Score'
col_explicativas = ['Study_Hours_Week', 'Attendance_Pct', 'Sleep_Hours_Day', 'Previous_Score']

df_limpo = df[['Final_Score'] + col_explicativas].dropna()

resul = {}

for col in col_explicativas:
    corr, p_valor = stats.pearsonr(df_limpo[col_alvo], df_limpo[col])

    resul[col] = {"corr": corr, 'p_valor' : p_valor}

df_resul = pd.DataFrame(resul).T

df_resul

# Temos que Study_Hours_Week é correlação mais forte entre todas. Com 0.5349, ou seja, quanto mais estuda pode ter impacto nas notas

# %%
# [ ] 18. Visualização de relações
# Crie gráficos de dispersão entre nota final e horas de estudo. 
# Adicione uma linha de tendência e procure padrões ou pontos influentes.
sns.set_theme(style='whitegrid')

plt.figure(figsize=(10,8))

sns.regplot(
    data=df,
    x='Study_Hours_Week',
    y='Final_Score',
    scatter_kws={'alpha':0.5,'color': 'royalblue'},
    line_kws={'color': 'crimson', 'linewidth': 2}
)

plt.title("Dispresão Horas de estudos vs Notas Final")
plt.xlabel("Horas de estudos por semana")
plt.ylabel("Notas Final")

plt.tight_layout()
plt.show()

dados = df[['Study_Hours_Week', 'Final_Score']].dropna()
res = stats.linregress(dados['Study_Hours_Week'], dados['Final_Score'])
print(f"Intercepto (B): {res.intercept:.2f} | Inclinação (Alfa): {res.slope:.2f} | R²: {res.rvalue**2:.4f}")
print(f"Y = {res.slope:.2f}X + {res.intercept:.2f} | X = Horas estudadas ")




# %%

import seaborn as sns
import matplotlib.pyplot as plt

cols = ['Final_Score', 'Study_Hours_Week', 'Attendance_Pct', 'Sleep_Hours_Day', 'Previous_Score']
sns.pairplot(df[cols].dropna(), kind='reg', plot_kws={'line_kws': {'color': 'crimson', 'linewidth': 1.5}, 'scatter_kws': {'alpha': 0.4}})
plt.suptitle('Relações Cruzadas entre Todas as Variáveis', y=1.02)
plt.show()

# %%

# [ ] 21. Pergunta de pesquisa
# Investigue: estudantes com frequência ≥ 80% têm notas finais maiores, em média, do que os demais?

# Aplicar Teste de Hipotese
# H0: a média das notas do grupo com frequência alta é MENOR ou igual à média do grupo com frequência baixa.
# H1: a média das notas do grupo com frequência alta é MAIOR que a média do grupo com frequência baixa.

# %%
grupo_frequencia_alta = df.loc[
    df['Attendance_Pct'] >= 80, 'Final_Score'
].dropna()

grupo_frequencia_baixa = df.loc[
    df['Attendance_Pct'] < 80, 'Final_Score'
].dropna()

#  Estatísticas descritivas
print("Frequência >= 80%")
print("Quantidade:", len(grupo_frequencia_alta))
print("Média:", grupo_frequencia_alta.mean())
print("Desvio-padrão:", grupo_frequencia_alta.std())

print("\nFrequência < 80%")
print("Quantidade:", len(grupo_frequencia_baixa))
print("Média:", grupo_frequencia_baixa.mean())
print("Desvio-padrão:", grupo_frequencia_baixa.std())


result_notas = stats.ttest_ind(grupo_frequencia_alta, grupo_frequencia_baixa, equal_var=False, alternative='greater')

print("\nEstatística t:", result_notas.statistic)
print("P-valor:", result_notas.pvalue)


# Foi realizado um teste t de Welch unilateral para comparar as notas finais dos estudantes com frequência ≥ 80%
# e daqueles com frequência abaixo de 80%.
# O resultado foi estatisticamente significativo t=3,63, p=0,000186. 
# Ao nível de significância de 5%, rejeitamos a hipótese nula e 
# encontramos evidências de que estudantes com frequência ≥ 80% apresentam notas
# médias superiores. Contudo, esse resultado demonstra uma associação, 
# não necessariamente uma relação causal.

# %%
# [ ] 22 - PERGUNTA PESQUISA: 

# Estudantes que dormem 7 horas ou mais por dia apresentam notas finais 
# médias maiores do que estudantes que dormem menos de 7 horas?

# População: Todos os estudantes
# Variavel Resposta: Final Score
# Grupos comparação: Alunos dormem >= 7 horas por dia vs Alunos dormem <= 7 horas
# H0: Estudantes que dormem 7 hrs ++ não tem notas melhores
# H1 Estudantes dormem 7 hrs ++ tem notas melhores

# GRUPOS
grupo_dorme_pouco = df.loc[df['Sleep_Hours_Day'] < 7, 'Final_Score'].dropna()

grupo_dorme_muito = df.loc[df['Sleep_Hours_Day'] >= 7, 'Final_Score'].dropna()

print("\n #### GRUPO DORME POUCO #####")
print("Quantidade", len(grupo_dorme_pouco))
print("Media: ", grupo_dorme_pouco.mean())
print("Desvio Padrao", grupo_dorme_pouco.std())

print("\n #### GRUPO DORME MUITO #####")
print("Quantidade", len(grupo_dorme_muito))
print("Media: ", grupo_dorme_muito.mean())
print("Desvio Padrao", grupo_dorme_muito.std())

# %%

resul_sono = stats.ttest_ind(grupo_dorme_muito, grupo_dorme_pouco,
                equal_var=False, 
                alternative='greater')

print("\nt: ", resul_sono.statistic)
print("P-valor: ", resul_sono.pvalue)
# %%
# Foi realizado teste T de welch unilateral para comparar as notas em dois grupos:
# GRUPO 1: Horas de sono maior igual a 7,
# GRUPO 2: Horas de sono menor a 7.
# O resultado foi estatisticamente significativo, com t = 1,47 desvio padrão
# Ou seja, indica uma associação onde grupos que dormem mais possue melhores notas
# Além disso, O teste realizado possui valor de significancia de 0.05
# Diante disso, seu pvalor foi: 0.07. Logo, não podemos rejeitar a hipotese nula
# (Estudantes com mais horas de sono tiram menores notas). 
# Portanto, nao foram encontradas evidencias estatisticas para concluir que estudantes
# Que dormem mais de 7 horas tiram notas melhores. Isso nao comprova que a media sejam iguais,
# nem permite concluir que a duração de sono impacta nas notas


## -=-=-=-==-=-=-=-=-=-==-=-=-=-==-==-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# %%
# INTERVALO DE CONFIANÇA

resultado = stats.ttest_ind(
    grupo_dorme_muito,
    grupo_dorme_pouco,
    equal_var=False
)

ic = resultado.confidence_interval(confidence_level=0.95)

limt_inf, limt_sup = ic

(limt_inf.item(), limt_sup.item())

# Estamos analisando media populacional de dois grupos:
# 1º Intervalo positivo (a,b) = Temos primeiro grupo com media superior
# 2º Intervalo negativo (-a,-b) = Temos segundo grupo com media superior
# 3º Intervalo contem zero (-a, b) = Nao podemos afirmar que existe uma diferença entre a media populacional]

# Nosso intervalo é (-0.608, 4,271) então não conseguimos descartar hip que as medias sejam iguais
# Isso significa que nao encontramos evidencia estatistica para afirmar que sao diferentes.
# Tudo isso se enquadra em EXISTE EVIDENCIA ESTATISTICA?

# ----------------=-=-=-=-=-----------=-=-=-=-=--=-
# %%

media_muito = grupo_dorme_muito.mean()
media_pouco = grupo_dorme_pouco.mean()

n1 = len(grupo_dorme_muito)
n2 = len(grupo_dorme_pouco)

dp1 = grupo_dorme_muito.std(ddof=1)
dp2 = grupo_dorme_pouco.std(ddof=1)

dp_combinado = np.sqrt(
    (n1 - 1) * dp1**2 + (n2 - 1) * dp2**2 / (n1 + n2 - 2)
    )

cohens_d = (media_muito - media_pouco) / dp_combinado

print("Média dorme muito:", media_muito)
print("Média dorme pouco:", media_pouco)
print("Desvio padrão combinado:", dp_combinado)
print("Cohen's d:", cohens_d)


