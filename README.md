# 🐆 Classificação de Comentários sobre Onças usando BERT

Este repositório contém todo o código utilizado para analisar, preparar e treinar modelos **BERT** com foco em classificação de comentários relacionados ao incidente das onças.

O projeto inclui:

* Exploração inicial dos dados
* Preparação e limpeza do dataset
* Treinamento de modelos BERT para múltiplas classes
* Avaliação e geração de métricas
* Salvar modelos e gráficos

---

## 📁 Estrutura do Projeto

```
├── explore_data.py        # Análise exploratória do dataset (colunas, valores, distribuição)
├── utils_dataset.py       # Pré-processamento, limpeza e split train/val/test
├── treinar_bert.py        # Script principal de treinamento BERT
├── models/
│   ├── onca/
│   │   ├── model/         # Modelo salvo (pytorch)
│   │   ├── tokenizer/     # Tokenizer salvo
│   │   ├── loss_onca.png  # Gráfico de perda
│   │   ├── errors.txt     # Amostras de erros do modelo
├── oncas_comentarios.xlsx # Arquivo original anotado com os rótulos
└── README.md
```

---

## 🧪 Scripts

### 🔍 **1. Exploração dos Dados** — `explore_data.py`

Este script exibe:

* Colunas existentes no dataset
* Distribuição das classes (onça, caseiro, fake news, etc.)
* Valores únicos por coluna
* Quantidade de textos vazios e duplicados

Use antes de iniciar qualquer treino para verificar se o dataset está correto.

### 🧹 **2. Preparação do Dataset** — `utils_dataset.py`

Funções responsáveis por:

* Remover textos vazios
* Remover duplicados
* Filtrar apenas classes válidas (negativo, neutro, positivo)
* Mapear classes → números
* Dividir em **70% treino / 15% validação / 15% teste**

### 🤖 **3. Treinamento BERT** — `train.py`

Script principal. Ele:

* Carrega o dataset via `utils_dataset.py`
* Tokeniza textos
* Treina um modelo BERT usando uma das classes
* Salva o modelo, tokenizador e gráficos
* Gera um relatório de classificação
* Exporta exemplos onde o modelo errou

---

## 📊 Exemplo de Resultados

O treino produz:

* **Gráfico de losses** ao longo das épocas
* **Relatório de classificação (precision, recall, F1)**
* Lista de erros do modelo para análise

Exemplo (classe: `onca`):

```
accuracy: 0.79
F1 neutro: 0.87
F1 negativo: 0.48
F1 positivo: 0.45
```

---

## 🚀 Como Executar

### 1. Instalar dependências (Já presentes no Colab)

```
pip install pandas scikit-learn torch transformers matplotlib openpyxl tqdm numpy
```

### 2. Executar o programa e escolher a label

```
!python train.py --label onca
!python train.py --label fake_news
!python train.py --label caseiro
!python train.py --label all
```

Modelos são salvos automaticamente em `models/<nome_da_classe>/`.

---

## 📚 Sobre o Dataset

As classes vêm da planilha **oncas_comentarios.xlsx** contendo:

* Comentários reais de redes sociais
* Anotações manuais indicando o sentimento ou característica do comentário

Classes usadas no modelo:

* `negativo`
* `neutro`
* `positivo`

* Obs: Foi alterado a coluna "fake news" para "fake_news".

---

## ▶️ Apresentação

* Link para apresentação: https://youtu.be/yod_Pr_E_dA?si=NB3r8gJIhTlvPL45

---
