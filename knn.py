import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

iris = load_iris()
X = iris.data
y = iris.target

# divide em 80% treinamento e 20% teste 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# treinamento
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

with open('resultados_knn.txt', 'w', encoding='utf-8') as f:
    print("--- RESULTADOS DO K-NN (K=3) ---", file=f)
    print(f"Acurácia: {accuracy_score(y_test, y_pred) * 100:.2f}%\n", file=f)
    print("Relatório de Classificação:\n", file=f)
    print(classification_report(y_test, y_pred, target_names=iris.target_names), file=f)

    # diferentes valores de K
    print("--- ACURÁCIA PARA DIFERENTES VALORES DE K ---", file=f)
    for k in [1, 3, 5, 7, 9, 11, 15]:
        modelo_k = KNeighborsClassifier(n_neighbors=k)
        modelo_k.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, modelo_k.predict(X_test_scaled))
        print(f"K = {k:2d} -> Acurácia: {acc * 100:.2f}%", file=f)
    
# gerar matriz de confusão k=3
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=iris.target_names, yticklabels=iris.target_names
)
plt.title('Matriz de Confusão do K-NN (K=3)')
plt.xlabel('Categoria Prevista pelo K-NN')
plt.ylabel('Categoria Real')
plt.tight_layout()
plt.savefig('matriz_confusao_knn.png', dpi=300)
plt.show()
