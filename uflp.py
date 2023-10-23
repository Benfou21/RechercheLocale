import math
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class UFLP():

    def __init__(self,instance_name : str) -> None:
        self.instance_name = instance_name
        self.load_instance(instance_name)

    def load_instance(self,instance_name : str) -> None:
        """Load an instance from a file and set the attributes of the class

        Args:
            instance_name (str): name of the instance to load, no extension needed and should be in the instances folder
        """
        self.main_stations_opening_cost = []
        self.main_stations_coordinates = []
        self.satellite_stations_connection_coordinates = []
        self.satellite_stations_connection_cost = []
        filename = "instances/"+instance_name+".txt"
        with open(filename, "r") as f:
            lines = f.readlines()
            self.n_main_station = int(lines[0].split(" ")[0])
            self.n_satellite_station = int(lines[0].split(" ")[1])
            for i in range(1,self.n_main_station+1):
                line = lines[i].split(" ")
                self.main_stations_opening_cost.append(float(line[2]))
                self.main_stations_coordinates.append((float(line[0]),float(line[1])))
            for i in range(self.n_main_station+1,self.n_main_station+self.n_satellite_station+1):
                line = lines[i].split(" ")
                self.satellite_stations_connection_coordinates.append((float(line[0]),float(line[1])))
        self.satellite_stations_connection_cost = [[self.coordinates_to_cost(self.main_stations_coordinates[i][0],self.main_stations_coordinates[i][1],self.satellite_stations_connection_coordinates[j][0],self.satellite_stations_connection_coordinates[j][1]) for j in range(self.n_satellite_station)] for i in range(self.n_main_station)]

    
    def coordinates_to_cost(self, x1 :float , y1 : float, x2 : float, y2 :float) -> float:
        """Calculate the cost of a connection between two stations

        Args:
            x1 (float): coordinate x of the first station
            y1 (float): coordinate y of the first station
            x2 (float): coordinate x of the second station
            y2 (float): coordinate y of the second station

        Returns:
            float: cost of the connection (distance between the two stations)
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
    
         

    def calcultate_cost(self,main_stations_opened: List[int], satellite_stations_association: list[int]) -> float:
        """Calculate the cost of a solution

        Args:
            main_stations_opened (List[int]): list of 0/1, 1 if the main station is opened, 0 otherwise
            satellite_stations_associations (list[int]): list of the main station associated to each satellite station

        Returns:
            float: cost of the solution
        """
        if sum(main_stations_opened) == 0: return math.inf
        opening_cost = sum([main_stations_opened[i]*self.main_stations_opening_cost[i] for i in range(self.n_main_station)])
        distance_cost = sum([self.satellite_stations_connection_cost[satellite_stations_association[i]][i] for i in range(len(satellite_stations_association))])
        return opening_cost+distance_cost
    
    def get_opening_cost(self,main_stations: int) -> float:
        """Get the opening cost of a main station

        Args:
            main_stations (int): index of the main station

        Returns:
            float: opening cost of the main station
        """
        return self.main_stations_opening_cost[main_stations]
    
    def get_association_cost(self,main_station: int,satellite_station: int) -> float:
        """Get the association cost of a satellite station to a main station

        Args:
            main_station (int): index of the main station
            satellite_station (int): index of the satellite station

        Returns:
            float: association cost of the satellite station to the main station
        """
        return self.satellite_stations_connection_cost[main_station][satellite_station]
    
    def show_solution(self,main_stations_opened: List[int], satellite_stations_association: list[int]) -> None:
        """Show the solution on a plot

        Args:
            main_stations_opened (List[int]): list of 0/1, 1 if the main station is opened, 0 otherwise
            satellite_stations_associations (list[int]): list of the main station associated to each satellite station
        """
        plt.figure(figsize=(10, 7))


        for i in range(self.n_main_station):
            plt.text(self.main_stations_coordinates[i][0], self.main_stations_coordinates[i][1], str(round(self.main_stations_opening_cost[i],2)), color='red')
            if main_stations_opened[i]:
                plt.scatter(self.main_stations_coordinates[i][0], self.main_stations_coordinates[i][1], marker='o', color='red', label=f'Gare principale {i+1}')
            else:
                plt.scatter(self.main_stations_coordinates[i][0], self.main_stations_coordinates[i][1], marker='s', color='red', label=f'Gare principale {i+1} (closed)')
        for j in range(self.n_satellite_station):
            plt.scatter(self.satellite_stations_connection_coordinates[j][0], self.satellite_stations_connection_coordinates[j][1], marker='o', color='blue', label=f'Gare satellite {j+1}')
        for i in range(self.n_satellite_station):
            for j in range(self.n_main_station):
                if satellite_stations_association[i] == j:

                    plt.plot([self.main_stations_coordinates[j][0], self.satellite_stations_connection_coordinates[i][0]], [self.main_stations_coordinates[j][1], self.satellite_stations_connection_coordinates[i][1]], color='black')
                    plt.text((self.main_stations_coordinates[j][0] + self.satellite_stations_connection_coordinates[i][0]) / 2, (self.main_stations_coordinates[j][1] + self.satellite_stations_connection_coordinates[i][1]) / 2, str(round(self.satellite_stations_connection_cost[j][i],2)), color='black')
                # pour afficher les couts de connexion des gares satellites aux gares principales
                """else:
                    plt.plot([self.main_stations_coordinates[j][0], self.satellite_stations_connection_coordinates[i][0]], [self.main_stations_coordinates[j][1], self.satellite_stations_connection_coordinates[i][1]], color='gray', linestyle='dashed')
                    plt.text((self.main_stations_coordinates[j][0] + self.satellite_stations_connection_coordinates[i][0]) / 2, (self.main_stations_coordinates[j][1] + self.satellite_stations_connection_coordinates[i][1]) / 2, str(round(self.satellite_stations_connection_cost[j][i],2)), color='gray')"""

        plt.title(f'Instance {self.instance_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.subplots_adjust(right=0.65)
        plt.savefig(f'instances/{self.instance_name}.png')
        plt.show()


    def solution_checker(self, main_stations_opened: List[int], satellite_stations_association: list[int]):
        """Check if a solution is valid
        
        Args:
            main_stations_opened (List[int]): list of 0/1, 1 if the main station is opened, 0 otherwise
            satellite_stations_associations (list[list[int]]): list of the main station associated to each satellite station

        Returns:
            bool: True if the solution is valid, False otherwise
            """

        if len(main_stations_opened) != self.n_main_station:
            print("Wrong solution: length of opened main stations does not match the number of main stations")
            return False
        if len(satellite_stations_association) != self.n_satellite_station:
            print("Wrong solution: length of associated satellite stations does not match the number of satellite stations")
            return False
        if sum(main_stations_opened) == 0:
            print("Wrong solution: no main station opened")
            return False
        
        for main_station in satellite_stations_association:

            if main_station < 0:
                print("Wrong solution: index of main station does not exist (< 0)")
                return False
            if main_station >= self.n_main_station:
                print("Wrong solution: index of main station does not exist (>= n)")
                return False
            if not main_stations_opened[main_station]:
                print("Wrong solution: assignation to a closed station")
                return False

        for state in main_stations_opened:

            if state not in [0,1]:
                print("Wrong solution: value different than 0/1 in main_stations_opened")
                return False
        
        return True
