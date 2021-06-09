#!/usr/bin/env python3

import json
import os

rotowire_dir = "/lnet/work/people/kasner/datasets/rotowire"

def extract_cities(j):
    cities = set()

    for record in j:
        home_city = record["home_city"]
        vis_city = record["vis_city"]

        cities.add(home_city)
        cities.add(vis_city)

    with open("cities.txt", "w") as f:
        for city in cities:
            f.write(city + "\n")

    return cities

    
if __name__ == "__main__":
    with open(os.path.join(rotowire_dir, "train.json")) as f:
        j = json.load(f)

    extract_cities(j)
