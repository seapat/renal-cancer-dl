import openslide
from monai.data import WSIReader
import time

base_path: str = (
    # "/media/sciobiome/DATA/sklein_tmp/Scans-QuPathProjekt-RCC-1Case-10062021/"
    "/data2/projects/DigiStrudMed_sklein/sample-case/Scans für QupathProjekt RCC (3 Fälle) 10062021/"   
)
case: str = "RCC-TA-011.001.023~M"
image_path: str = base_path + case + ".svs"
print(f"Image path: {image_path} \n")

target_level: int = 0
print(f"Target level: {target_level} \n")
#######################################################

start1 = time.time()

openslide_image = openslide.OpenSlide(image_path)
big = openslide_image.read_region(
    location=(0, 0), level=target_level, size=openslide_image.level_dimensions[0]
)

end1 = time.time()
print(f"Time = {end1 - start1}")
print(
    f'''Openslide : image shape = {big.size}
    openslide Props = {openslide_image.properties}
    Info = {big.info}'''
)

#######################################################

start2 = time.time()

image_reader = WSIReader(backend="openslide")
wsi = image_reader.read(image_path)
img_data, meta_data = image_reader.get_data(wsi, level=target_level)

end2 = time.time()
print(f"Time = {end2 - start2}")
print(
    f'''Monai(openslide) : shape = {img_data.shape}
    Path = {image_reader.get_file_path(wsi)}'''
)

# RESUTLS:
# openslide 668 secs for level 0
# monai 724 secs for level 0