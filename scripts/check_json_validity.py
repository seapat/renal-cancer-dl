# This script validates the geojson files in the given directory.

# check if array is empty
# check if unnamed coordinates exist
# Check if name "tissue" exists
from glob import glob
import json
import os


search_path: str = "/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/all_data/"
files: list = glob(os.path.join(search_path, "*.json"))

empty_files: list = []
missing_tissue: list = []
unnamed_annotations: list = []
# no_name: bool = False
for file in files:
    no_tissue:bool = True
    json_arr: list = json.load(open(file))
    no_name: bool = False
    unnamed_annotation_count: int = 0
    # Empty file
    if len(json_arr) == 0:
        empty_files.append(os.path.basename(file))
        print(f"Empty file {os.path.basename(file)}")
    else:
        for json_obj in json_arr:
            if "properties" in json_obj.keys():
                if "classification" in json_obj["properties"].keys():
                    # no_tissue implies wheteher we've seen a Tissue annotation in the current file so far
                    if "name" in json_obj["properties"]["classification"].keys():
                        if json_obj["properties"]["classification"]["name"] == "Tissue":
                            no_tissue = False
                else:
                    # The current object has no name ...
                    no_name = True
                    # ... but has coordinates
                    if "coordinates" in json_obj["geometry"].keys():
                        unnamed_annotation_count += 1
                        unnamed_annotations.append(os.path.basename(file))
                        
                        # The current object also has no coordinates (unlikely)
                        if json_obj["geometry"]["coordinates"] == []:
                            print(f"empty coordinates in {os.path.basename(file)}")
                    else:
                        print(f"No name and no coords in {os.path.basename(file)}")

        # Check no_tissue at the end of the current file
        if no_tissue:
            missing_tissue.append(os.path.basename(file))
            print(f"No tissue in {os.path.basename(file)}")
    if unnamed_annotation_count > 0:
        print(f"unamed annotations in {os.path.basename(file)}: {unnamed_annotation_count}")


# []
print(f"{len(empty_files)} Empty files: \n\t {empty_files}") 
# no "name": "Tissue" in json
print(f"{len(missing_tissue)} Missing tissues: \n\t {missing_tissue}")
# valid coords but no name in properties
print(f"{len(unnamed_annotations)} Unnamed annotations: \n {unnamed_annotations}") 
print("\n")

incorrect_files: frozenset = frozenset(file.removesuffix(".json") for file in empty_files + missing_tissue + unnamed_annotations)
all_json: frozenset = frozenset(os.path.basename(file.removesuffix(".json")) for file in glob(os.path.join("/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/all_data/", "*.json")))
curr_tissues: frozenset = frozenset(os.path.basename(file.removesuffix("-Tissue.tif")) for file in glob(os.path.join("/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/all_data/", "*-Tissue.tif")))
not_rcc: frozenset = frozenset(os.path.basename(file.removesuffix(".json")) for file in  glob(os.path.join("/data2/projects/DigiStrudMed_sklein/DigiStrucMed_Braesen/all_data/", "M*.json")))

print(f"No. of incorrect json files: {len(incorrect_files)}")
print(f"No. of json files in dir: {len(all_json)}")
print(f"No. masks of Tissue in dir: {len(curr_tissues)}")
print(f"No. of Non-RCC jsons: {len(not_rcc)}")

missing_tissues: frozenset = all_json - curr_tissues - not_rcc
missing_cases: frozenset = all_json - curr_tissues - not_rcc - incorrect_files

print(f"No. of RCC tissues that are missing: {len(missing_tissues)}")
# print(f"{(missing_tissues)}")
print(f"No. of Tissues that are missing completely: {len(missing_cases)}")
print(f"{(missing_cases)}")

''' shortened output:

4 Empty files: 
17 Missing tissues: 
13 Unnamed annotations: 

incorrect_files: 22
all_json: 774
curr_tissues: 713
missing_tissues: 23
missing_cases: 2

'''

'''
4 Empty files: 
17 Missing tissues: 
13 Unnamed annotations: 

No. of incorrect json files: 22
No. of json files in dir: 957
No. masks of Tissue in dir: 891
No. of Non-RCC jsons: 38
No. of RCC tissues that are missing: 28
No. of Tissues that are missing completely: 7
frozenset({'RCC-TA-035.001~B+A', 'RCC-TA-036.001~B', 'RCC-TA-196.019~B', 'RCC-TA-051.001~B', 'RCC-TA-007.001~B', 'RCC-TA-196.041~B', 'RCC-TA-185.001_2nd~C'})
'''