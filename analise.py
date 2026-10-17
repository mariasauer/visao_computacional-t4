import pandas as pd
from sklearn.datasets import load_iris

# carrega dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# coluna com as categorias
df['categoria'] = pd.Categorical.from_codes(iris.target, iris.target_names)

with open('resultados_analise.txt', 'w', encoding='utf-8') as f:
    print("--- FREQUÊNCIA DE CADA CATEGORIA ---", file=f)
    print(df['categoria'].value_counts(), file=f)
    print("\n" + "+"*50 + "\n", file=f)

    print("--- ESTATÍSTICAS GERAIS ---", file=f)
    estatisticas_gerais = pd.DataFrame({
        'Média': df.iloc[:, 0:4].mean(),
        'Desvio Padrão': df.iloc[:, 0:4].std(),
        'Moda': df.iloc[:, 0:4].mode().iloc[0]
    })
    print(estatisticas_gerais.round(2), file=f)
    print("\n" + "+"*50 + "\n", file=f)

    print("--- ESTATÍSTICAS POR CATEGORIA ---", file=f)
    media_desvio_cat = df.groupby('categoria', observed=False).agg(['mean', 'std']).round(2)
    print("Média e Desvio Padrão:", file=f)
    print(media_desvio_cat, file=f)

    print("\nModa por Categoria:", file=f)
    moda_cat = df.groupby('categoria', observed=False).agg(lambda x: x.mode().iloc[0])
    print(moda_cat, file=f)
