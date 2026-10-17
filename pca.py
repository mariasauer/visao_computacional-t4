import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# dataset
iris = load_iris()
X = iris.data
y = iris.target
nomes_categorias = iris.target_names

# padroniza os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['categoria'] = pd.Categorical.from_codes(y, nomes_categorias)

variancia = pca.explained_variance_ratio_ * 100
pesos = pd.DataFrame(pca.components_, columns=iris.feature_names, index=['PC1', 'PC2'])

with open('resultados_pca.txt', 'w', encoding='utf-8') as f:
    print("--- RESULTADOS DO PCA ---", file=f)
    print(f"Variância explicada pelo PC1: {variancia[0]:.2f}%", file=f)
    print(f"Variância explicada pelo PC2: {variancia[1]:.2f}%", file=f)
    print(f"Informação total retida em apenas 2D: {sum(variancia):.2f}%\n", file=f)

    print("--- PESO DAS VARIÁVEIS ORIGINAIS NOS COMPONENTES ---", file=f)
    print(pesos.round(2), file=f)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca, x='PC1', y='PC2', 
    hue='categoria', style='categoria', s=90
)
plt.title(f'PCA do Dataset Iris (Retendo {sum(variancia):.1f}% da Variância Total)')
plt.xlabel(f'Componente Principal 1 ({variancia[0]:.1f}%)')
plt.ylabel(f'Componente Principal 2 ({variancia[1]:.1f}%)')
plt.tight_layout()
plt.savefig('pca_iris.png', dpi=300)
plt.show()
