import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

def create_digits()->np.ndarray:
    digits = []
    digit_0 = np.array([
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    ])
    digit_1 = np.array([
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    ])
    digit_2 = np.array([
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    ])
    digit_3 = np.array([
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    ])
    digit_4 = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    ])
    digit_5 = np.array([
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    ])
    digit_6 = np.array([
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    ])
    digit_7 = np.array([
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ])
    digit_8 = np.array([
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    ])
    digit_9 = np.array([
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    ])
    digits.append(digit_0)
    digits.append(digit_1)
    digits.append(digit_2)
    digits.append(digit_3)
    digits.append(digit_4)
    digits.append(digit_5)
    digits.append(digit_6)
    digits.append(digit_7)
    digits.append(digit_8)
    digits.append(digit_9)
    return np.array(digits)


def plot_matriz(matr):
  plt.imshow(matr, cmap='gray', interpolation='nearest')
  plt.colorbar()
  plt.show()

def hamming_distance(mat1, mat2):
  if mat1.shape != mat2.shape:
        raise ValueError("As matrizes devem ter o mesmo comprimento")

  return np.sum(mat1 != mat2)

def select_top_n_hamming_distances(hamming_dist, top_n, decreasing=False):
  allCombinations = list(combinations(range(10), top_n))#Gerar combinações de todos os valores entre 0 e 9, ou seja, (0,1,2); (0,1,3); (0,1,4); ...; (6,8,9); (7,8,9)
  best_comb = None
  best_comb_value = 0
  if decreasing == True:
    best_comb_value = np.inf

  for comb in allCombinations:
    current_sum = hamming_dist[(comb[0], comb[1])]
    current_sum += hamming_dist[(comb[0], comb[2])]
    current_sum += hamming_dist[(comb[1], comb[2])]
    if decreasing == False and current_sum > best_comb_value:
      best_comb = comb
      best_comb_value = current_sum
    elif decreasing == True and current_sum < best_comb_value:
      best_comb = comb
      best_comb_value = current_sum
  return best_comb


def add_noise(pattern, noise_level):
    noisy_pattern = pattern.copy()
    num_noisy_bits = int(noise_level * len(pattern))
    flip_indices = random.sample(range(len(pattern)), num_noisy_bits) # Selecionar randomicamente todos os "num_noisy_bits" a serem invertidos
    noisy_pattern[flip_indices] = 1 - noisy_pattern[flip_indices]  # Inverter bits
    return noisy_pattern

def pattern_to_vector(pattern):
    return np.array([1 if num == 1 else 0 for row in pattern for num in row])

def pattern_to_vector(pattern):
    return np.array([1 if num == 1 else 0 for row in pattern for num in row])

class HopfieldNetwork:
  def __init__(self, size):
    self.size = size
    self.weights = np.zeros((size, size))

  def train(self, patterns:list[np.ndarray]):
    """Treina a rede com os padrões binários."""
    for pattern in patterns:
        p = np.where(pattern == 0, -1, pattern) # Converter para {-1, 1}
        p = p.reshape(-1, 1)
        self.weights += np.outer(p, p)
    np.fill_diagonal(self.weights, 0)  # Zerar a diagonal

  def recall(self, pattern, max_iter=10, synchronous = True):
    """Executa o processo de recuperação do padrão."""
    state = pattern.copy()
    state = np.where(state == 0, -1, state)
    # print("recalling")
    for _ in range(max_iter):
        new_state = self.update(state,synchronous)
        if np.array_equal(state,new_state): break #convergiu
        state = new_state
        # print(self.energy(state))
    return state
  
  def update(self, state, synchronous=True):
        """
        Update the network state using synchronous or asynchronous rule.
        """
        if synchronous:
            # All neurons update at once
            new_state = np.sign(self.weights @ state)
            new_state[new_state == 0] = 1  # Avoid zero states
            return new_state
        else:
            # Asynchronous update (one neuron at a time)
            new_state = np.copy(state)
            for i in range(self.size):
                new_state[i] = np.sign(np.dot(self.weights[i], new_state))
                if new_state[i] == 0:
                    new_state[i] = 1
            return new_state
  
  def energy(self, state):
        """
        Computes the energy of the current state.
        Lower energy indicates stable states.
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))

#================#
# código da main #
#================#
if __name__ == '__main__':  
  digits = create_digits()

  # for i in range(10):
  #   plot_matriz(digits[i])

  # plot_matriz(add_noise(digits[0], 0.1)) #Exemplo do dígito 0 com 10% de ruído

  all_hamming_distances = {}
  for i in range(10):
    for j in range(10):
      if i >= j:
        continue
      print(f"Distância de Hamming entre {i} e {j}:")
      all_hamming_distances[(i,j)] = hamming_distance(digits[i], digits[j])
      print(hamming_distance(digits[i], digits[j]))
    print("\n")
    
  top_3_best = select_top_n_hamming_distances(all_hamming_distances, 3)
  print(f"3 padrões com maior distância de Hamming: {top_3_best}")

  top_3_worst = select_top_n_hamming_distances(all_hamming_distances, 3, True)
  print(f"3 padrões com menor distância de Hamming: {top_3_worst}")

  top_5_best = select_top_n_hamming_distances(all_hamming_distances, 5)
  print(f"5 padrões com maior distância de Hamming: {top_5_best}")

  top_7_best = select_top_n_hamming_distances(all_hamming_distances, 7)
  print(f"7 padrões com maior distância de Hamming: {top_7_best}")

  #digits_3_best = [pattern_to_vector(digits[d]) for d in top_3_best]
  digits_3_best = [pattern_to_vector(digits[top_3_best[0]]),pattern_to_vector(digits[top_3_best[1]]), pattern_to_vector(digits[top_3_best[2]])]
  #print(digits_3_best)
  hopfield = HopfieldNetwork(size=len(digits_3_best[0]))
  hopfield.train(digits_3_best)

  noise_levels = [0, 5, 10, 20, 30, 40, 50]
  accuracy_results_3_best = []

  for noise in noise_levels:
    correct_recognitions = 0
    total_tests = 5 * len(digits_3_best)  # 5 testes por padrão
    # print(total_tests)
    for pattern in digits_3_best:
      for _ in range(5):
        noisy_pattern = add_noise(pattern, noise / 100)
        recalled_pattern = hopfield.recall(noisy_pattern)#, synchronous=False)
        recalled_pattern = np.where(recalled_pattern == -1,0,recalled_pattern)
        if np.array_equal(recalled_pattern, pattern):
          correct_recognitions += 1
        # else:
        #   plot_matriz(np.reshape(np.where(recalled_pattern!=pattern,1,0),(12,10)))
        #   plot_matriz(np.array(recalled_pattern).reshape(12, 10))
        #   plot_matriz(np.array(pattern).reshape(12, 10))
    accuracy = (correct_recognitions / total_tests) * 100
    accuracy_results_3_best.append(accuracy)

  digits_3_worst = [pattern_to_vector(digits[d]) for d in top_3_worst]
  hopfield = HopfieldNetwork(size=len(digits_3_worst[0]))
  hopfield.train(digits_3_worst)

  accuracy_results_3_worst = []

  for noise in noise_levels:
    correct_recognitions = 0
    total_tests = 5 * len(digits_3_worst)  
    for pattern in digits_3_worst:
      for _ in range(5):
        noisy_pattern = add_noise(pattern, noise / 100)
        recalled_pattern = hopfield.recall(noisy_pattern)#, synchronous=False)
        recalled_pattern = np.where(recalled_pattern == -1,0,recalled_pattern)
        if np.array_equal(recalled_pattern, pattern):
          correct_recognitions += 1
        # else:
        #   plot_matriz(np.array(recalled_pattern).reshape(12, 10))
    accuracy = (correct_recognitions / total_tests) * 100
    accuracy_results_3_worst.append(accuracy)


  digits_5_best = [pattern_to_vector(digits[d]) for d in top_5_best]

  hopfield = HopfieldNetwork(size=len(digits_5_best[0]))
  hopfield.train(digits_5_best)

  accuracy_results_5_best = []

  for noise in noise_levels:
    correct_recognitions = 0
    total_tests = 5 * len(digits_5_best)  # 5 testes por padrão
    for pattern in digits_5_best:
      for _ in range(5):
        noisy_pattern = add_noise(pattern, noise / 100)
        recalled_pattern = hopfield.recall(noisy_pattern)#, synchronous=False)
        recalled_pattern = np.where(recalled_pattern == -1,0,recalled_pattern)
        if np.array_equal(recalled_pattern, pattern):
          correct_recognitions += 1
        # else:
        #   plot_matriz(np.array(recalled_pattern).reshape(12, 10))
    accuracy = (correct_recognitions / total_tests) * 100
    accuracy_results_5_best.append(accuracy)

  digits_7_best = [pattern_to_vector(digits[d]) for d in top_7_best]

  hopfield = HopfieldNetwork(size=len(digits_7_best[0]))
  hopfield.train(digits_7_best)

  noise_levels = [0, 5, 10, 20, 30, 40, 50]
  accuracy_results_7_best = []

  for noise in noise_levels:
    correct_recognitions = 0
    total_tests = 5 * len(digits_7_best)  # 5 testes por padrão
    for pattern in digits_7_best:
      for _ in range(5):
        noisy_pattern = add_noise(pattern, noise / 100)
        recalled_pattern = hopfield.recall(noisy_pattern)#, synchronous=False)
        recalled_pattern = np.where(recalled_pattern == -1,0,recalled_pattern)
        if np.array_equal(recalled_pattern, pattern):
          correct_recognitions += 1
        #else:
          #plot_matriz(np.array(recalled_pattern).reshape(12, 10))
    accuracy = (correct_recognitions / total_tests) * 100
    accuracy_results_7_best.append(accuracy)


  digits_10 = [pattern_to_vector(digits[d]) for d in range(10)]

  hopfield = HopfieldNetwork(size=len(digits_10[0]))
  for _ in range(10000):
    hopfield.train(digits_10)

  accuracy_results_10 = []

  for noise in noise_levels:
    correct_recognitions = 0
    total_tests = 5 * len(digits_10)  # 5 testes por padrão
    for pattern in digits_10:
      for _ in range(5):
        noisy_pattern = add_noise(pattern, noise / 100)
        recalled_pattern = hopfield.recall(noisy_pattern)#, synchronous=False)
        recalled_pattern = np.where(recalled_pattern == -1,0,recalled_pattern)
        if np.array_equal(recalled_pattern, pattern):
          correct_recognitions += 1
        # else:
        #   print("detectou:")
        #   plot_matriz(np.array(recalled_pattern).reshape(12, 10))
        #   print("era:")
        #   plot_matriz(np.array(pattern).reshape(12, 10))
    accuracy = (correct_recognitions / total_tests) * 100
    accuracy_results_10.append(accuracy)

  # Plotar gráfico de desempenho
  plt.plot(noise_levels, accuracy_results_3_best, marker='s', linestyle='-', label="3 Padrões mais próximos")
  plt.plot(noise_levels, accuracy_results_3_worst, marker='o', linestyle='-', label="3 Padrões Mais distantes")
  plt.plot(noise_levels, accuracy_results_5_best, marker='*', linestyle='-', label="5 Padrões mais próximos")
  plt.plot(noise_levels, accuracy_results_7_best, marker='h', linestyle='-', label="7 Padrões mais próximos")
  plt.plot(noise_levels, accuracy_results_10, marker='^', linestyle='-', label="10 Padrões")
  plt.xlabel("Nível de Ruído (%)")
  plt.ylabel("Percentual de Acerto (%)")
  plt.title("Desempenho da Rede Hopfield na Recuperação de Padrões")
  plt.legend()
  plt.grid()
  plt.show()