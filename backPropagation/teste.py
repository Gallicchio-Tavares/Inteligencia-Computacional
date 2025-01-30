import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

# Funções para mapear dígitos com ruídos diferentes
def create_bit_maps()->np.ndarray:
    return np.array([
        [0, 1, 0, 0,
         1, 1, 0, 0,
         0, 1, 0, 0,
         0, 1, 0, 0,
         1, 1, 1, 0],
        
        [0, 1, 1, 0,
         1, 0, 0, 1,
         0, 0, 1, 0,
         0, 1, 0, 0,
         1, 1, 1, 1],
        
        [1, 1, 1, 0,
         0, 0, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1,
         1, 1, 1, 0],
        
        [1, 0, 1, 0,
         1, 0, 1, 0,
         1, 1, 1, 1,
         0, 0, 1, 0,
         0, 0, 1, 0],
        
        [1, 1, 1, 1,
         1, 0, 0, 0,
         1, 1, 1, 0,
         0, 0, 0, 1,
         1, 1, 1, 0],
        
        [0, 1, 1, 0,
         1, 0, 0, 0,
         1, 1, 1, 0,
         1, 0, 0, 1,
         0, 1, 1, 0],
        
        [1, 1, 1, 1,
         0, 0, 0, 1,
         0, 0, 1, 0,
         0, 1, 0, 0,
         0, 1, 0, 0],
        
        [0, 1, 1, 0,
         1, 0, 0, 1,
         0, 1, 1, 0,
         1, 0, 0, 1,
         0, 1, 1, 0],
        
        [0, 1, 1, 0,
         1, 0, 0, 1,
         0, 1, 1, 1,
         0, 0, 0, 1,
         1, 1, 1, 1],

        [0, 1, 1, 0,
         1, 0, 0, 1,
         1, 0, 0, 1,
         1, 0, 0, 1,
         0, 1, 1, 0]]).T

def add_noise(bitmap, noise_level=1):
    noisy_bitmap = bitmap.copy()
    for _ in range(noise_level):
        i = np.random.randint(0, bitmap.shape[0])
        noisy_bitmap[i] = 1 - noisy_bitmap[i]  # Inverte o valor (0 -> 1 ou 1 -> 0)
    return noisy_bitmap

def plot_digit(digit):
    plt.imshow(digit, cmap="winter", interpolation="nearest")
    plt.axis("off")
    plt.show()

# Criação dos mapas com diferentes níveis de ruído
def generate_noisy_bitmaps(bitmaps:np.ndarray, noise_levels):
    transposed_bitmaps=bitmaps.T
    noisy_bitmaps = {lvl: {} for lvl in noise_levels}
    for key, bitmap in enumerate(transposed_bitmaps):
        for noise_level in noise_levels:
            noisy_bitmaps[noise_level][key] = add_noise(bitmap, noise_level)
    return noisy_bitmaps

# Função para treinar a rede neural
def train_network(input_data, target_data, hidden_neurons, learning_rate, epochs):
    model = Sequential([
        Dense(hidden_neurons, activation="tanh", input_dim=input_data.shape[1]),
        Dense(target_data.shape[1], activation="linear")
    ])
    optimizer = SGD(learning_rate=learning_rate, momentum=0.4)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    model.fit(input_data, target_data, epochs=epochs, verbose=1)
    return model

if __name__ == '__main__':
    # Preparo dos dados
    bitmaps = create_bit_maps()
    noisy_bitmaps = generate_noisy_bitmaps(bitmaps, noise_levels=[0, 1, 2, 3])

    # Vetorização dos dígitos
    # P = np.array([bitmap.flatten() for bitmap in bitmaps.values()])+
    T = np.eye(len(bitmaps))  # Matriz identidade como saída esperada

    # Treinamento da rede
    hidden_neurons = 15
    learning_rate = 0.1
    epochs = 10000

    model = train_network(bitmaps, T, hidden_neurons, learning_rate, epochs)

    # Testes com ruído
    # Salvar resultados num .txt de acordo com a combinacao de informacoes (ex: rede 15 com lr 0.9 e momentum 0.0)
    filename = f"saida.txt"
    with open(filename, "w", encoding="utf-8") as f:
        for noise_level, noisy_data in noisy_bitmaps.items():
            P_noisy = np.array([bitmap.flatten() for bitmap in noisy_data.values()])
            predictions = model.predict(P_noisy)
            f.write(f"Resultados com {noise_level} ruídos:\n")
            f.write(np.array2string(predictions, precision=4, suppress_small=True))
            f.write("\n\n")
    print(f"Resultados salvos em {filename}\n")