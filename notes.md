# Docs

## Replace all strings in Qupath projects

- `file:.*[\/]` will match all files declaredi .qpproj files (vscode regex) to replace with proper paths
- Afterwards we have to fix the path leading to the .qpproj files themselves (optional)
- the run `./QuPath script ./renal-cancer-dl/groovy/GeoJSON.groovy -p [...].qpproj`

## Edge cases, cases with problems:

- insufficient Paitent data, eg survival time can not  be  calculated
- Qupath annotations:
    - missing completely (empty array)
    - no "Tissue" annotation
        - it is missing completely
        - there is a 'mysterious' Annotation w/o a name ( which could be anything, not necessarily tissue)

The following Files are affected:

'RCC-TA-120.001~B', 'RCC-TA-038.016~B', 'RCC-TA-108.001~B', 'RCC-TA-121.001~B', 'RCC-TA-131.001~B', 'RCC-TA-123.001~B','RCC-TA-104.016~B', 'RCC-TA-035.016~B', 'RCC-TA-196.041~B', 'RCC-TA-115.001~B', 'RCC-TA-124.001~E', 'RCC-TA-126.001~B', 'RCC-TA-100.016~B', 'RCC-TA-092.001~D', 'RCC-TA-098.001~B', 'RCC-TA-036.016~B', 'RCC-TA-116.001~B', 'RCC-TA-107.001~B', 'RCC-TA-196.019~B', 'RCC-TA-127.001~B', 'RCC-TA-118.001~B', 'RCC-TA-122.016~B', 'RCC-TA-196.019~B', 'RCC-TA-196.041~B', 'RCC-TA-110.001~E+A', 

Some Stats: 
4 Empty files
13 Unnamed annotations and 17 Missing tissues (these overlap)
No. of incorrect json files: 22 (sum of the cases above)

No. of json files in dir: 774   (`*.json`)
No. masks of Tissue in dir: 713 (`*-Tissue.tif`)
No. of Non-RCC jsons: 38        (`*M*.json`)
No. of RCC tissues that are missing: 23 (`*.json`- `*-Tissue.tif` - `*M*.json`)
No. of Tissues that are missing completely: 2 [`*.json`- `*-Tissue.tif` - `*M*.json` - Union(4, 13, 17)]

The 2 missing completely are both  from case 196, this case lacks follow up dates or status of survival. It would be useless to us either way.

Some Image are cut off, this leads to annotations outside of the image boundary, we reset the annotation to the boundary of the image.
Sometimes tiles are partly outside the image because of how close the tissue is to the border. In those cases we extract a smaller patch and pad it with zeroes afterwards.

## Survival Times: Ties

Out of the 141 cases leftover after fitlering them on the tabluar data, there is one tie for the survival times. Conretely for the cases 093 and 114. Hence we chose to adopt breslow tie handling to deal with this.
