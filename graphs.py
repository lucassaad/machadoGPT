import matplotlib.pyplot as plt

# Exemplo de dados (substitua pelos seus)
texto = [
    (4.9878,4.9864),
(2.0784,2.0704),
(1.6161,1.6121),
(1.4600,1.4608),
(1.3775,1.3905),
(1.3381,1.3622),
(1.2983,1.3221),
(1.2772,1.3073),
(1.2551,1.2912),
(1.2426,1.2881),
(1.2220,1.2766),

]
poesia = [
   (4.9343,4.9370),
(2.1742,2.2297),
(1.6848,1.8349),
(1.4772,1.6698),
(1.3358,1.6371),
(1.2342,1.6288),
(1.1314,1.6298),
(1.0321,1.6752),
(0.9468,1.7230),
(0.8526,1.7548),
(0.7665,1.8169),

]
teatro = [
(4.7182,4.7016),
(1.8908,1.8722),
(1.3785,1.4421),
(1.1226,1.3467),
(0.9143,1.3666),
(0.7178,1.4511),
(0.5372,1.5550),
(0.4006,1.6925),
(0.3021,1.8097),
(0.2306,1.9214),
(0.1844,2.0565),
]

# Desempacotando cada linha em duas listas: valores de val1 e val2
texto_train, texto_val = zip(*texto)
poesia_train, poesia_val = zip(*poesia)
teatro_train, teatro_val = zip(*teatro)

# Criando os 2 gráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico 1: val1 das 3 linhas
ax1.plot(texto_train, color='red', label='texto')
ax1.plot(poesia_train, color='blue', label='poesia')
ax1.plot(teatro_train, color='green', label='teatro')
ax1.set_title('Train Loss')
ax1.legend()

# Gráfico 2: val2 das 3 linhas
ax2.plot(texto_val, color='red', label='texto')
ax2.plot(poesia_val, color='blue', label='poesia')
ax2.plot(teatro_val, color='green', label='teatro')
ax2.set_title('Val Loss')
ax2.legend()

plt.tight_layout()
plt.savefig('grafico_loss_com_dropout_collab.png', dpi=150)