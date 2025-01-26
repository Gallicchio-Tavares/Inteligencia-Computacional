import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_network(input_shape, hidden_units, output_units):
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(input_shape,), activation='tanh'))
    model.add(Dense(output_units, activation='linear'))
    return model

def train_network(model, X, y, epochs=10000, learning_rate=0.1, momentum=0.0):
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model

def network15():
    from digits import bit_maps
    
    num1, num2, num3, num4, num5, num6, num7, num8, num9, num0 = bit_maps()
    
    # Preparando os dados de entrada e saída
    P = np.array([num1.flatten(), num2.flatten(), num3.flatten(), num4.flatten(),
                  num5.flatten(), num6.flatten(), num7.flatten(), num8.flatten(),
                  num9.flatten(), num0.flatten()])  # Removido .T
    T = np.eye(10)
    
    # Criando a rede neural
    model = create_network(input_shape=P.shape[1], hidden_units=15, output_units=10)
    
    # Treinamento
    model = train_network(model, P, T, epochs=10000, learning_rate=0.1, momentum=0.0)
    print("Treinamento 1 concluído.")
    
    model = train_network(model, P, T, epochs=10000, learning_rate=0.4, momentum=0.0)
    print("Treinamento 2 concluído.")
    
    model = train_network(model, P, T, epochs=10000, learning_rate=0.9, momentum=0.0)
    print("Treinamento 3 concluído.")
    
    model = train_network(model, P, T, epochs=10000, learning_rate=0.1, momentum=0.4)
    print("Treinamento 4 concluído.")
    
    model = train_network(model, P, T, epochs=10000, learning_rate=0.9, momentum=0.4)
    print("Treinamento 5 concluído.")
    
    # Testes
    print("Início dos testes das redes.")
    A = model.predict(P)
    print("Teste 0 ruídos:")
    print(A)
    
    print("Fim network15.")