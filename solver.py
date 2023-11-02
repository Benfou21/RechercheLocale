from uflp import UFLP
import random
from typing import List, Tuple
import numpy as np
import time
import math



def solve(problem : UFLP)-> Tuple[List[int], List[int]]:

    """
    Votre implementation, doit resoudre le probleme via recherche locale.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations satellites au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """

    best_sol = generate_random_sol(problem)
    start_time = time.time()
    
    while(time.time() - start_time < 2*60):  #2min
        sol = solve_1(problem)
        if sol[0] < best_sol[0] :
            best_sol = sol
    
    return best_sol[1]
        

def solve_1(problem: UFLP) -> Tuple[List[int], List[int]]:

    
    sol = generate_random_sol(problem)
    
   
    
    start_time = time.time()
    
    while(time.time() - start_time < 2*60):  #2min
        V = generate_neigh_2(problem,sol,1)
        V_validated = validated_function(V,sol[0])

        if V_validated != []:
            selected_sol = selection(V_validated)
            
            if selected_sol[0] < sol[0]:
                sol = selected_sol
        else:
            break
    
    return sol





def solve_simulated_a(problem: UFLP) -> Tuple[List[int], List[int]]:

   
    sol = generate_random_sol(problem)
    start_time = time.time()
    
    temperature = 1000  # valeur initiale
    cooling_rate = 0.995  # taux de refroidissement
    
    while(time.time() - start_time < 2*60):  # 2min
        V = generate_neigh_1(problem, sol, 1)
        V_validated = validated_function_degradation(V, sol[0], temperature)

        if len(V_validated) > 0:
            selected_sol = selection_with_probability(V_validated, sol[0], temperature)
            sol = selected_sol  # Ici, nous acceptons la nouvelle solution, même si elle est pire

        temperature *= cooling_rate  # réduire la température

    return sol



def generate_random_sol(problem : UFLP):
    # Ouverture aléatoire des stations principales
    main_stations_opened = [random.choice([0,1]) for _ in range(problem.n_main_station)]

    # Si, par hasard, rien n'est ouvert, on ouvre une station aléatoirement
    if sum(main_stations_opened) == 0:
        main_stations_opened[random.choice(range(problem.n_main_station))] = 1

    # Association aléatoire des stations satellites aux stations principales ouvertes
    indices = [i for i in range(len(main_stations_opened)) if main_stations_opened[i] == 1]
    satellite_station_association = [random.choice(indices) for _ in range(problem.n_satellite_station)]

    lists = [main_stations_opened, satellite_station_association]
    
    cost = problem.calcultate_cost(main_stations_opened=lists[0], satellite_stations_association=lists[1])

    return [cost,lists]

def heuristic_initial_solution(problem: UFLP):
    # Trouver la station principale avec le coût d'ouverture le plus bas
    cheapest_main_station = min(range(problem.n_main_station), key=lambda i: problem.get_opening_cost(i))
    
    # Ouvrir cette station
    main_stations_opened = [0] * problem.n_main_station
    main_stations_opened[cheapest_main_station] = 1
    
    # Associer chaque station satellite à cette station principale
    satellite_station_association = [cheapest_main_station] * problem.n_satellite_station
    
    # Calculer le coût de cette solution initiale
    cost = problem.calcultate_cost(main_stations_opened=main_stations_opened, satellite_stations_association=satellite_station_association)
    
    return [cost, [main_stations_opened, satellite_station_association]]


def heuristic_initial_solution_2(problem: UFLP):
    # Trié les gares pp par cout
    list_station = list(range(problem.n_main_station)).sort(key=lambda i: problem.get_opening_cost(i))
    
    # Ouvrir 1er station
    main_stations_opened = [0] * problem.n_main_station
    main_stations_opened[list_station[0] ] = 1
    # Associer chaque station satellite à cette station principale
    
    satellite_station_association = best_connection(problem,  main_stations_opened)
    # Calculer le coût de cette solution initiale
    cost = problem.calcultate_cost(main_stations_opened=main_stations_opened, satellite_stations_association=satellite_station_association)
    
    return [cost, [main_stations_opened, satellite_station_association], list_station]



def best_connection(problem: UFLP,  main_stations_opened):
    
    
    # Initialiser l'association des stations satellites comme vide
    satellite_station_association = [-1] * problem.n_satellite_station
    
    # Parcourir chaque station satellite pour l'associer à la meilleure station principale
    for s in range(problem.n_satellite_station):
        best_cost_for_s = float('inf')
        best_main_station_for_s = -1
        
        for m in range(problem.n_main_station):
            # Calculer le coût si cette station satellite était associée à cette station principale
            temp_stations_opened = main_stations_opened.copy()
            temp_stations_opened[m] = 1  # Assumer que la station principale m est ouverte
            temp_station_association = satellite_station_association.copy()
            temp_station_association[s] = m  # Associer la station satellite s à m
            
            temp_cost = problem.calcultate_cost(main_stations_opened=temp_stations_opened, satellite_stations_association=temp_station_association)
            
            # Mettre à jour le meilleur coût et la meilleure station principale pour cette station satellite
            if temp_cost < best_cost_for_s:
                best_cost_for_s = temp_cost
                best_main_station_for_s = m
                
        # Mettre à jour la solution globale avec la meilleure association pour cette station satellite
        main_stations_opened[best_main_station_for_s] = 1
        satellite_station_association[s] = best_main_station_for_s
        
    
    return satellite_station_association

# def generate_neigh_1(problem, sol, nb_changes):
#     print(sol)
#     print("Generate")
#     V = []


    
#     list_selected = []
    
#     for n in range(nb_changes) :

#         satellite_station = [x for x in range(problem.n_satellite_station) if x not in list_selected]
#         print(satellite_station)
#         s = random.choice(satellite_station)
#         print(s)
        
#         main_stations = [x for x in range(problem.n_main_station) if x != sol[1][1][s]] #sol[1][1] = satellites
#         print(sol[1][1][s])
#         print(main_stations)
#         sol[1][1][s] = random.choice(main_stations)

#         # Maj gare pp
#         sol[1][0] = [0] * (len(sol[1][0]) )  #sol[1][0] = principales
#         for satellite in sol[1][1]:
#             sol[1][0][satellite] = 1
        
#         print(problem.solution_checker(main_stations_opened=sol[1][0], satellite_stations_association=sol[1][1]))
#         if problem.solution_checker(main_stations_opened=sol[1][0], satellite_stations_association=sol[1][1]) :
#             cost = problem.calcultate_cost(main_stations_opened=sol[1][0], satellite_stations_association=sol[1][1])
            
#             V.append([cost,sol])

#     return V


from itertools import combinations


#Mvt = changement de connexion
def generate_neigh_1(problem, sol, nb_changes):
    V = []
    
    # Générer toutes les combinaisons possibles de `nb_changes` stations satellites
    all_combinations = combinations(range(problem.n_satellite_station), nb_changes)
    
    for combination in all_combinations:
        # Parcourir toutes les stations principales pour chaque combinaison de stations satellites
        for m_list in combinations(range(problem.n_main_station), nb_changes):
            new_sol = [list(x) for x in sol[1]]  # Créer une copie profonde de la solution actuelle
            
            skip = False  # Un flag pour sauter des solutions redondantes
            for s, m in zip(combination, m_list):
                if m == sol[1][1][s]:
                    skip = True
                    break
                new_sol[1][s] = m  # Associer la station satellite 's' à la station principale 'm'
            
            if skip:
                continue
                
            # Mettre à jour les stations principales ouvertes
            new_sol[0] = [0] * len(new_sol[0])
            for satellite in new_sol[1]:
                new_sol[0][satellite] = 1
                
            if problem.solution_checker(main_stations_opened=new_sol[0], satellite_stations_association=new_sol[1]):
                cost = problem.calcultate_cost(main_stations_opened=new_sol[0], satellite_stations_association=new_sol[1])
                
                V.append([cost, new_sol])

    return V

#Mvt = Ouverture/fermeture d'une gare PP, puis best connection
from typing import List, Tuple
import copy
import itertools

def generate_neigh_2(problem, sol: Tuple[int, List[int]], nb_changes: int) -> List[Tuple[int, List[int]]]:
    V = []
    
    # Générer toutes les combinaisons possibles de `nb_changes` stations principales
    for pp_combination in combinations(range(problem.n_main_station), nb_changes):
        for open_close in itertools.product([0, 1], repeat=nb_changes):  # 0 pour fermer, 1 pour ouvrir
            new_sol = [list(x) for x in sol[1]]  # Créer une copie profonde de la solution actuelle
            
            # Appliquer les ouvertures/fermetures
            for pp, oc in zip(pp_combination, open_close):
                new_sol[0][pp] = oc
                
            # Assurer qu'au moins une gare principale est ouverte
            if sum(new_sol[0]) == 0:
                continue
            
            # Utiliser la fonction best_connection pour trouver les meilleures connexions
            new_sol[1] = best_connection(problem, new_sol[0])
            
            # Vérifier la validité de la nouvelle solution et calculer son coût
            if problem.solution_checker(main_stations_opened=new_sol[0], satellite_stations_association=new_sol[1]):
                cost = problem.calcultate_cost(main_stations_opened=new_sol[0], satellite_stations_association=new_sol[1])
                V.append([cost, new_sol])
                
    return V

def validated_function(V,actual_cost):

    return [v for v in V if v[0] < actual_cost] 


def validated_function_degradation(V, actual_cost, temperature):
    validated = []
    for v in V:
        delta_cost = v[0] - actual_cost
        if delta_cost < 0:
            validated.append(v)
        elif random.random() < math.exp(-delta_cost / temperature):
            validated.append(v)
    return validated

def selection(V):
    
    return min(V, key=lambda x: x[0])

def first_selection(V):
    
    return V[0]

def random_selection(V):

    return random.choice(V)


def selection_with_probability(V, actual_cost, temperature):
    exp_values = [math.exp(-(v[0] - actual_cost) / temperature) for v in V]
    sum_exp_values = sum(exp_values)

    probabilities = [exp_value / sum_exp_values for exp_value in exp_values]

    return random.choices(V, weights=probabilities, k=1)[0]