from cucim import CuImage
from monai.data import WSIReader
import time

base_path: str = (
    "/home/sklein/Thesis/Scans-QuPathProjekt-RCC(1Case)-10062021/"
    # "/media/sciobiome/DATA/sklein_tmp/Scans-QuPathProjekt-RCC-1Case-10062021/"
    # "/data2/projects/DigiStrudMed_sklein/sample-case/Scans für QupathProjekt RCC (3 Fälle) 10062021/"
)
case: str = "RCC-TA-011.001.023~M"
image_path: str = base_path + case + ".svs"
print(f"Image path: {image_path} \n")

target_level: int = 0
print(f"Target level: {target_level} \n")
#######################################################

start1 = time.time()

cu_image = CuImage(image_path)
big = cu_image.read_region(level=target_level) #, device="cuda")

end1 = time.time()
print(f"Time = {end1 - start1}")
print(f"LoadImage : shape = {big.shape} \n Path = {big.path}")

#######################################################

start2 = time.time()

image_reader = WSIReader(backend="cucim")# , device="cuda")
wsi = image_reader.read(image_path)
img_data, meta_data = image_reader.get_data(wsi, level=target_level)

end2 = time.time()
print(f"Time = {end2 - start2}")
print(
    f"Monai(cucim) : shape = {img_data.shape} \n Path = {image_reader.get_file_path(wsi)}"
)

# RESULT:
# WITH CUDA:
#   ON ALIENWARE
#       both crash but with different errors depending if level 0 or if level is higher
# WITHOUT CUDA
#   ON ALIENWARE
#       Cucim: 285 secs at level 0 
#       Monai: 291 secs at level 0
#   ON norm
#       Cucim: 440 secs at level 0
#       Monai: 461 secs at level 0