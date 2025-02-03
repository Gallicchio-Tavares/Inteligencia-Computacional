import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore

# Funções para mapear dígitos com ruídos diferentes
def create_bit_maps():
    bitmaps = {
        0: np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 0, 0, 1],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0]]),
        
        1: np.array([[0, 1, 0, 0],
                     [1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0],
                     [1, 1, 1, 0]]),
        
        2: np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [1, 1, 1, 1]]),
        
        3: np.array([[1, 1, 1, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [1, 1, 1, 0]]),
        
        4: np.array([[1, 0, 1, 0],
                     [1, 0, 1, 0],
                     [1, 1, 1, 1],
                     [0, 0, 1, 0],
                     [0, 0, 1, 0]]),
        
        5: np.array([[1, 1, 1, 1],
                     [1, 0, 0, 0],
                     [1, 1, 1, 0],
                     [0, 0, 0, 1],
                     [1, 1, 1, 0]]),
        
        6: np.array([[0, 1, 1, 0],
                     [1, 0, 0, 0],
                     [1, 1, 1, 0],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0]]),
        
        7: np.array([[1, 1, 1, 1],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 1, 0, 0]]),
        
        8: np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0]]),
        
        9: np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [0, 1, 1, 1],
                     [0, 0, 0, 1],
                     [1, 1, 1, 1]])
        
    }
    return bitmaps

def add_noise(bitmap, noise_level=1):
    noisy_bitmap = bitmap.copy()
    for _ in range(noise_level):
        i, j = np.random.randint(0, bitmap.shape[0]), np.random.randint(0, bitmap.shape[1])
        noisy_bitmap[i, j] = 1 - noisy_bitmap[i, j]  # Inverte o valor (0 -> 1 ou 1 -> 0)
    return noisy_bitmap

def generate_noisy_bitmaps(bitmaps, noise_levels):
    noisy_bitmaps = {lvl: {} for lvl in noise_levels}
    for key, bitmap in bitmaps.items():
        for noise_level in noise_levels:
            noisy_bitmaps[noise_level][key] = add_noise(bitmap, noise_level)
    return noisy_bitmaps

# Função para treinar a rede neural
def train_network(input_data, target_data, hidden_neurons, learning_rate, momentum, epochs):
    model = Sequential([
        Dense(hidden_neurons, activation="tanh", input_dim=input_data.shape[1]), # usa tansig (camada oculta)
        Dense(target_data.shape[1], activation="linear") # usa purelin na camada de saida
    ])
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    model.fit(input_data, target_data, epochs=epochs, verbose=1)
    return model

bitmaps = create_bit_maps() 
noisy_bitmaps = generate_noisy_bitmaps(bitmaps, noise_levels=[0, 1, 2, 3]) #definindo os ruidos de 0 a 3

# Vetorização dos dígitos, ou seja, transformar na matriz 20x10 como diz no trab
P = np.array([bitmap.flatten() for bitmap in bitmaps.values()])
T = np.eye(len(bitmaps))  # matriz identidade, que é a saída esperada

networks = [
    {"name": "network15", "hidden_neurons": 15},
    {"name": "network25", "hidden_neurons": 25},
    {"name": "network35", "hidden_neurons": 35}
]

lr_momentum_combinations = [ # combinacoes de learning rate (lr) e momentum
    (0.1, 0.0),
    (0.4, 0.0),
    (0.9, 0.0),
    (0.1, 0.4),
    (0.9, 0.4)
]

# Treinamento e teste das redes em si
for network in networks:
    for lr, momentum in lr_momentum_combinations:
        print(f"Treinando {network['name']} com lr={lr}, momentum={momentum}")
        model = train_network(P, T, network["hidden_neurons"], lr, momentum, epochs=10000)
        
        # Testes com ruído, 0 a 3 ruídos. noise_level eh definido mais acima no codigo
        results = {}
        for noise_level, noisy_data in noisy_bitmaps.items():
            P_noisy = np.array([bitmap.flatten() for bitmap in noisy_data.values()])
            predictions = model.predict(P_noisy)
            results[noise_level] = predictions
        
        # Salvar resultados num .txt de acordo com a combinacao de informacoes (ex: rede 15 com lr 0.9 e momentum 0.0)
        filename = f"{network['name']}_lr{lr}_momentum{momentum}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for noise_level, predictions in results.items():
                f.write(f"Resultados com {noise_level} ruídos:\n")
                f.write(np.array2string(predictions, precision=4, suppress_small=True, max_line_width=np.inf))
                f.write("\n\n")
        print(f"Resultados salvos em {filename}\n")
