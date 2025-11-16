# utils_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path, label_col):
    df = pd.read_excel(path)

    # Remove textos vazios ou faltantes
    df = df.dropna(subset=['comment_text'])

    # Remover duplicados
    df = df.drop_duplicates(subset=['comment_text'])

    # Classes esperadas
    valid_classes = ['negativo', 'neutro', 'positivo']

    df = df[df[label_col].isin(valid_classes)]

    # Mapear para números
    label_map = {'negativo': 0, 'neutro': 1, 'positivo': 2}
    df['label'] = df[label_col].map(label_map)

    # Divisão: 70 / 15 / 15
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

    return train_df, val_df, test_df, label_map
