import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['categoria'] = pd.Categorical.from_codes(iris.target, iris.target_names)

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(
    data=df, x='sepal length (cm)', y='sepal width (cm)', 
    hue='categoria', style='categoria', s=80, ax=axes[0]
)
axes[0].set_title('Relação: Comprimento vs Largura da Sépala')

# Gráfico das Pétalas
sns.scatterplot(
    data=df, x='petal length (cm)', y='petal width (cm)', 
    hue='categoria', style='categoria', s=80, ax=axes[1]
)
axes[1].set_title('Relação: Comprimento vs Largura da Pétala')

plt.tight_layout()
plt.savefig('relacao_sepalas_petalas.png', dpi=300) # Salva a imagem na pasta T4
plt.show()

g = sns.pairplot(df, hue='categoria', height=2.2, diag_kind='kde')
g.fig.suptitle('Distribuição das Categorias nas 4 Dimensões', y=1.02)
plt.savefig('distribuicao_4_dimensoes.png', dpi=300) # Salva a imagem na pasta T4
plt.show()

with open('resultados_graficos.txt', 'w', encoding='utf-8') as f:
    print("--- CORRELAÇÃO: SÉPALA (Comprimento vs Largura) ---", file=f)
    for cat, group in df.groupby('categoria', observed=False):
        corr = group['sepal length (cm)'].corr(group['sepal width (cm)'])
        print(f"{cat}: {corr:.2f}", file=f)
        
    print("\n--- CORRELAÇÃO: PÉTALA (Comprimento vs Largura) ---", file=f)
    for cat, group in df.groupby('categoria', observed=False):
        corr = group['petal length (cm)'].corr(group['petal width (cm)'])
        print(f"{cat}: {corr:.2f}", file=f)
