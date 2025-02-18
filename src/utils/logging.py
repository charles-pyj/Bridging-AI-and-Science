import pandas as pd
import pickle
import json

def read_file(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects

def read_json(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def parse_json(string):
    #print(string)
    try:
        dict = json.loads(string)
    except:
        return {"Scientific problems (in short)": "Not mentioned", "Scientific problems (in detail)": "Not mentioned","AI methods (in short)": "N/A","AI Employment": "N/A"}
    return dict

def read_coords(filename):
    x = []
    y = []
    with open('C:\\Users\\charl\\Research\\forseer\\LargeVis\\results\\2014_2023.txt', 'r') as file:
    # Read each line in the file
        next(file)
        for line in file:
            # Strip any leading/trailing whitespace characters
            line = line.strip()
            
            # Split the line into a list of strings on the whitespace
            coords = line.split()
            
            # Convert strings to floats and append to respective lists
            x.append(float(coords[0]))
            y.append(float(coords[1]))
    print(len(x))
    return x,y
# Check the contents of the list

def parse_descriptor_all(mesh):
    meshes = []
    mesh = eval(mesh)
    for i in mesh.values():
        meshes.append(i['descriptor_name'])
    return meshes

def parse_descriptor_major(mesh):
    meshes = []
    mesh = eval(mesh)
    for i in mesh.values():
        if(i['major_topic'] == True):
            meshes.append(i['descriptor_name'])
    return meshes