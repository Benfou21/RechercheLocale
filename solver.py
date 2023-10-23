from uflp import UFLP
import random
from typing import List, Tuple
import numpy as np
import time
def solve(problem: UFLP) -> Tuple[List[int], List[int]]:




    """
    Votre implementation, doit resoudre le probleme via recherche locale.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations satellites au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """

    
    sol = generate_random_sol(problem)
    start_time = time.time()
    
    while(time.time() - start_time < 2*60):  #2min
        V = generate_neigh_1(problem,sol,1)
        V_validated = validated_function(V,sol[0])

        if V_validated != []:
            selected_sol = selection(V_validated)
            
            if selected_sol[0] < sol[0]:
                sol = selected_sol
        else:
            break
    
    return sol[1]




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


def validated_function(V,actual_cost):

    return [v for v in V if v[0] < actual_cost] 

def selection(V):
    print(V)
    return min(V, key=lambda x: x[0])