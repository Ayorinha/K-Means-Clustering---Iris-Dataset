# K-Means Clustering — Iris Dataset

> Demonstração de aprendizado não supervisionado com K-Means, avaliação por índice de silhueta e visualização dos clusters.

## Problema

Explorar como um algoritmo de clustering pode identificar grupos em dados sem utilizar rótulos durante o treinamento.

## Solução

O projeto aplica K-Means ao dataset Iris, buscando três clusters a partir das medidas de sépala e pétala. A qualidade do agrupamento é avaliada pelo **Silhouette Score** e os resultados são visualizados em duas dimensões.

## Funcionalidades

- Carregamento do dataset Iris.
- Preparação das variáveis numéricas.
- Agrupamento com K-Means.
- Avaliação por índice de silhueta.
- Visualização dos clusters e centróides.
- Script Python executável localmente.

## Stack

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Como executar

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python kmeans_clustering.py
```

## Observação metodológica

O Iris é utilizado como dataset de demonstração. O objetivo do projeto é mostrar o processo de clustering e avaliação, não construir um modelo de produção.

## Próximos passos

- Tornar `k` configurável.
- Comparar diferentes inicializações.
- Adicionar testes automatizados.
- Comparar métricas de clustering.
- Documentar experimentos e resultados.

## Autor

**Anderson Leon Ayora**  
Data Scientist | AI Engineer | Data Architect
