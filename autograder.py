import math
from uflp import UFLP
import solver

print("***********************************************************")
print("[INFO] Start the autograding: train network design")
print("***********************************************************\n\n")

instances = ["instance_A_4_6","instance_B_25_50","instance_C_50_75"]
scores_randoms = [300,5000,15000]#ordre de grandeur random
scores_secrets = [187.06494521726978,1757.9431857942498,3383.65597024572]
instances = [UFLP(i) for i in instances]
scores_beaten = [[],[]]
has_failed = False
for i in range(len(instances)):
    print("***********************************************************")
    print("[INFO] autograding: instance",instances[i].instance_name)
    try:
        main_stations_opened, satellite_station_association = solver.solve(instances[i])
        score = instances[i].calcultate_cost(main_stations_opened, satellite_station_association)
        print("[INFO] RUN: passed")
    except Exception as e:
        print("[INFO] RUN: failed :",e)
        score = math.inf
        has_failed = True
    print("[INFO] score: ",score)
    print("[INFO] Random player beaten ("+str(scores_randoms[i])+"):", score < scores_randoms[i])
    print("[INFO] Secret player beaten ("+str(scores_secrets[i])+"):", score <= scores_secrets[i] if i == 0 else score < scores_secrets[i])
    scores_beaten[0].append(score < scores_randoms[i])
    scores_beaten[1].append(score <= scores_secrets[i] if i == 0 else score < scores_secrets[i])
    print("***********************************************************\n\n")

print("***********************************************************")
print("[INFO] autograding: summary")
if has_failed:
    print("[INFO] RUN: failed, 0/10")
else:
    
    print("[INFO] RUN: passed, >0/10")
    if sum(scores_beaten[0]) < 3:
        print("[INFO] Random player beaten: failed, <5/10")
        print("[INFO] Hint: vérifiez que votre recherche locale cherche à minimiser le coût, vérifiez que la fonction de voisinage est correcte, que la recherche locale s'arrête bien quand il n'y a plus d'amélioration possible")
    else:
        print("[INFO] Random player beaten: passed, >=5/10")
        if (sum(scores_beaten[1]) < 3):
            print("[INFO] Secret player beaten: failed, <8/10")
            print("[INFO] Hint: Utilisez des metaheuristiques comme le recuit simulé ou le restart")
        else:
            print("[INFO] Secret player beaten: passed, >=8/10")
print("***********************************************************\n\n")
    
