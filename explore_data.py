import pandas as pd
import sys

# Configurar encoding para UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Ler o arquivo Excel
df = pd.read_excel('oncas_comentarios.xlsx')

print('Colunas:', df.columns.tolist())
print('\nShape:', df.shape)

# Verificar colunas de rótulos (parecem ser: onca, caseiro, fake news, ironia, notícia)
label_cols = ['onca', 'caseiro', 'fake news', 'ironia', 'notícia']
print('\nColunas de rótulos encontradas:', [col for col in label_cols if col in df.columns])

# Verificar valores nas colunas de rótulos
print('\nValores únicos em cada coluna de rótulo:')
for col in label_cols:
    if col in df.columns:
        unique_vals = df[col].dropna().unique()
        print(f'{col}: {len(unique_vals)} valores únicos - {unique_vals}')

# Verificar comment_text
if 'comment_text' in df.columns:
    print(f'\nTotal de textos: {len(df)}')
    print(f'Textos vazios: {df["comment_text"].isna().sum()}')
    print(f'Textos duplicados: {df["comment_text"].duplicated().sum()}')
    
# Verificar quantos exemplos temos para cada classe
print('\nContagem de exemplos por classe:')
for col in label_cols:
    if col in df.columns:
        count = df[col].notna().sum()
        print(f'{col}: {count} exemplos')

