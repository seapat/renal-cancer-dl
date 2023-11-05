/* groovylint-disable VariableTypeRequired */
import qupath.lib.images.writers.ome.OMEPyramidWriter

// --- PARAMETERS ---------------------------------------------------------------------------------------------------------------------------
double downsample = 4d
int tile_size = 512
def format = "ome.tif"
// Define output path (relative to project)
def testDir = buildFilePath('/data2/projects/DigiStrudMed_sklein/test_annotations')
mkdirs(testDir)

def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
// ----------------------------------------------------------------------------------------------------------------------------------------------------

// --- Define binary masks --------------------------------------------------------------------------------------------------------------------

def tissueMaskServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK)
  .addLabel('Tissue', 1)
  .downsample(downsample)
  .multichannelOutput(false)
  .build()
new OMEPyramidWriter.Builder(tissueMaskServer)
  .tileSize(tile_size)
  .channelsInterleaved()
  .downsamples(1, 4, 16, 64, 256)
  .scaledDownsampling(1d, downsample)
  .parallelize()
  .losslessCompression()
  .allZSlices() 
  .build()
  .writePyramid(buildFilePath(testDir, name + '-Tissue_singleChannel.' + format))

def tissueMaskServer2 = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK)
  .addLabel('Tissue', 1)
  .downsample(downsample)
  .multichannelOutput(true)
  .build()
new OMEPyramidWriter.Builder(tissueMaskServer2)
  .downsamples(1, 4, 16, 64, 256)
  .scaledDownsampling(1d, downsample)
  .tileSize(tile_size)
  .parallelize()
  .losslessCompression()
  .allZSlices()
  .build()
  .writePyramid(buildFilePath(testDir, name + '-Tissue_multiChannel.' + format))

def fullMaskServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK)
  .addLabel('Tissue', 1)
  .addLabel('Tumor', 2)
  .addLabel('Tumor_vital', 2)
  .addLabel('diffuse tumor growth in soft tissue', 2)
  .addLabel('Angioinvasion', 2)
  .addLabel('Tumor_necrosis', 2)
  .addLabel('Tumor_regression', 2)
  .downsample(downsample)
  .multichannelOutput(false)
  .build()
new OMEPyramidWriter.Builder(fullMaskServer)
  .downsamples(1, 4, 16, 64, 256)
  .scaledDownsampling(1d, downsample)
  .tileSize(tile_size)
  .parallelize(32)
  .losslessCompression()
  .allZSlices()
  .build()
  .writePyramid(buildFilePath(testDir, name + '-Full_singleChannel.' + format))
    
def fullMaskServer2 = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK)
  .addLabel('Tissue', 1)
  .addLabel('Tumor', 2)
  .addLabel('Tumor_vital', 2)
  .addLabel('diffuse tumor growth in soft tissue', 2)
  .addLabel('Angioinvasion', 2)
  .addLabel('Tumor_necrosis', 2)
  .addLabel('Tumor_regression', 2)
  .downsample(downsample)
  .multichannelOutput(true)
  .build()
new OMEPyramidWriter.Builder(fullMaskServer2)
  .downsamples(1, 4, 16, 64, 256)
  .scaledDownsampling(1d, downsample)
  .tileSize(tile_size)
  .parallelize(32)
  .losslessCompression()
  .allZSlices()
  .build()
  .writePyramid(buildFilePath(testDir, name + '-Full_multiChannel.' + format))
// ----------------------------------------------------------------------------------------------------------------------------------------------------

// --- Write images to pyramidal files ------------------------------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------------------------------------------------------------------------
