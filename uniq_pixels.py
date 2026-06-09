import pandas as pd

# carregar o CSV
df = pd.read_csv("white_pixels.csv")

# remover duplicatas (todas as colunas)
df_unico = df.drop_duplicates()

# salvar novo arquivo
df_unico.to_csv("white_pixels_uniq.csv", index=False)

print("Pronto! Arquivo salvo como saida_unicos.csv")