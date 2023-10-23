from uflp import UFLP
import random
from typing import List, Tuple

def solve(problem: UFLP) -> Tuple[List[int], List[int]]:
    """
    Retourne une solution aléatoire au probleme : à battre pour avoir la moyenne.

    Args:
        problem (UFLP): L'instance du probleme à résoudre

    Returns:
        Tuple[List[int], List[int]]: 
        La premiere valeur est une liste représentant les stations principales ouvertes au format [0, 1, 0] qui indique que seule la station 1 est ouverte
        La seconde valeur est une liste représentant les associations des stations satellites au format [1 , 4] qui indique que la premiere station est associée à la station pricipale d'indice 1 et la deuxieme à celle d'indice 4
    """

    # Ouverture aléatoire des stations principales
    main_stations_opened = [random.choice([0,1]) for _ in range(problem.n_main_station)]

    # Si, par hasard, rien n'est ouvert, on ouvre une station aléatoirement
    if sum(main_stations_opened) == 0:
        main_stations_opened[random.choice(range(problem.n_main_station))] = 1

    # Association aléatoire des stations satellites aux stations principales ouvertes
    indices = [i for i in range(len(main_stations_opened)) if main_stations_opened[i] == 1]
    satellite_station_association = [random.choice(indices) for _ in range(problem.n_satellite_station)]

    return main_stations_opened, satellite_station_association

