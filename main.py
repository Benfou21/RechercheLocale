import argparse
import time

import random_solver
import solver
from uflp import UFLP

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Instances parameters
    parser.add_argument('--agent', type=str, default='random')
    parser.add_argument('--infile', type=str, default='instance_A_4_6')
    # if --preview is present, preview is at true else false
    parser.add_argument('--preview', action='store_true')


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    uflp = UFLP(args.infile)

    print("***********************************************************")
    print("[INFO] Start the solving: train network design")
    print("[INFO] instance: %s" % args.infile)
    print("[INFO] agent: %s" % args.agent)
    print("[INFO] number of main stations: %s" % (uflp.n_main_station))
    print("[INFO] number of satellite stations: %s" % (uflp.n_satellite_station))
    print("***********************************************************")

    start_time = time.time()


    if args.agent == "random":
        main_stations_opened, satellite_station_association = random_solver.solve(uflp)
    elif args.agent == "advanced":
        # Your nice agent
        main_stations_opened, satellite_station_association =  solver.solve(uflp)

    else:
        raise Exception("This agent does not exist")
    



    solving_time = round((time.time() - start_time) / 60,2)

    print("***********************************************************")
    print("[INFO] Solution obtained")
    print("[INFO] Execution time : %s minutes" % solving_time)
    print("[INFO] Main stations opened :",main_stations_opened)
    print("[INFO] Satellite station association :",satellite_station_association)
    print("[INFO] Penality obtained (value to minimize) : %s" % uflp.calcultate_cost(main_stations_opened, satellite_station_association))
    print("[INFO] Sanity check passed : %s" % uflp.solution_checker(main_stations_opened, satellite_station_association))
    print("***********************************************************")

    if args.preview:
        uflp.show_solution(main_stations_opened, satellite_station_association)