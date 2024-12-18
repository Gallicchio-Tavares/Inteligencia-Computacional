import numpy as np
import matplotlib.pyplot as plt

def f6(x, y):
    numerador = (np.sin(np.sqrt(x**2 + y**2)))**2 - 0.5
    denominador = (1 + 0.001 * (x**2 + y**2))**2
    return 0.5 - numerador / denominador

def initialize_population(tamanho_pop, limite_crom): #limite de cromossomos e tamanho da populacao
    return np.random.uniform(limite_crom[0], limite_crom[1], size=(tamanho_pop, 2))

def evaluate_population(populacao):
    return np.array([f6(x, y) for x, y in populacao])

def fitness_windowing(evals): # tal do windowing
    min_eval = np.min(evals)
    return evals - min_eval

def fitness_normalization(evals): #normalizacao linear 
    min_eval, max_eval = np.min(evals), np.max(evals)
    return 1 + 99 * (evals - min_eval) / (max_eval - min_eval)

def tournament_selection(populacao, fitness, tournament_size=3):
    selected_indices = np.random.choice(len(populacao), tournament_size)
    best_index = selected_indices[np.argmax(fitness[selected_indices])]
    return populacao[best_index]

def crossover(pai1, pai2, rate_crossover): #parte do crossover do AG
    if np.random.rand() < rate_crossover:
        alfa = np.random.rand()
        filho1 = alfa * pai1 + (1 - alfa) * pai2
        filho2 = alfa * pai2 + (1 - alfa) * pai1
        return filho1, filho2
    return pai1, pai2

def mutate(individuo, rate_mutacao, limite_crom): #faz as mutacoes
    if np.random.rand() < rate_mutacao:
        gene = np.random.randint(len(individuo))
        individuo[gene] = np.random.uniform(limite_crom[0], limite_crom[1])
    return individuo

def genetic_algorithm(tamanho_pop, geracoes, limite_crom, rate_crossover, rate_mutacao, 
                      fitness_method='evaluation', elitism=True, steady_state=False, no_duplicates=False): #o nosso AG
    populacao = initialize_population(tamanho_pop, limite_crom)
    best_solutions = []

    for gen in range(geracoes):
        evals = evaluate_population(populacao)

        if fitness_method == 'windowing':
            fitness = fitness_windowing(evals)
        elif fitness_method == 'normalization':
            fitness = fitness_normalization(evals)
        else:
            fitness = evals

        if elitism:
            melhor_idx = np.argmax(fitness)
            melhor_individuo = populacao[melhor_idx]
            melhor_valor = evals[melhor_idx]
        
        nova_pop = []

        while len(nova_pop) < tamanho_pop:
            pai1 = tournament_selection(populacao, fitness)
            pai2 = tournament_selection(populacao, fitness)
            filho1, filho2 = crossover(pai1, pai2, rate_crossover)
            # vai mutar
            filho1 = mutate(filho1, rate_mutacao, limite_crom)
            filho2 = mutate(filho2, rate_mutacao, limite_crom)

            nova_pop.extend([filho1, filho2])

        populacao = np.array(nova_pop[:tamanho_pop])# n exceder o tamanho

        if steady_state and no_duplicates:
            populacao = np.unique(populacao, axis=0)
            while len(populacao) < tamanho_pop:
                new_individual = np.random.uniform(limite_crom[0], limite_crom[1], size=2)
                populacao = np.vstack([populacao, new_individual])

        if elitism: #aki vai inserir o melhor individuo
            populacao[np.random.randint(len(populacao))] = melhor_individuo
        # e daí salva o melhor resultado da geracao atual
        best_solutions.append(melhor_valor if elitism else np.max(evals))

    return best_solutions

def run_experiments():
    tamanho_pop = 100
    geracoes = 40
    limite_crom = [-100, 100]
    rate_crossover = 0.8
    rate_mutacao = 0.05

    metodos_fitness = ['evaluation', 'windowing', 'normalization']
    resultado = {}

    for metodo in metodos_fitness:
        resultado[metodo] = []
        for exp in range(20): # 20 experimentos
            print(f"Rodando experimento {exp+1} para aptidão: {metodo}")
            best_solutions = genetic_algorithm(
                tamanho_pop, geracoes, limite_crom,
                rate_crossover, rate_mutacao, fitness_method=metodo, elitism=False
            )
            resultado[metodo].append(best_solutions)

    plt.figure(figsize=(10, 6)) #! parte do gráfico em si
    for metodo in metodos_fitness:
        media_melhor = np.mean(resultado[metodo], axis=0)
        plt.plot(range(geracoes), media_melhor, label=f'Aptidão: {metodo.capitalize()}')

    plt.xlabel('Gerações')
    plt.ylabel('Média das Melhores Soluções')
    plt.title('Comparação dos Métodos de Aptidão (Sem Elitismo)')
    plt.legend()
    plt.show()

    return resultado

if __name__ == "__main__":
    run_experiments()
