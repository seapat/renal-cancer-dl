import qupath.lib.images.writers.ome.OMEPyramidWriter

// --- PARAMETERS ---------------------------------------------------------------------------------------------------------------------------
double downsample = 1d
int tileSize = 512
String format = 'ome.tif'
// Define output path (relative to project)
def annoDir = buildFilePath('/data2/projects/DigiStrudMed_sklein/AnnotationMasks')
mkdirs(annoDir)
// ----------------------------------------------------------------------------------------------------------------------------------------------------

// --- Define binary masks --------------------------------------------------------------------------------------------------------------------
def imageData = getCurrentImageData()

def fullMaskServer = new LabeledImageServer.Builder(imageData)
  // Order is important here - late labels overwrite previous ones in case of overlaps
  // Specify background label (usually 0 or 255)
  .backgroundLabel(0, ColorTools.BLACK)
  // Each class requires a name and a number
  .addLabel('Tissue', 1)
  .addLabel('Tumor', 2)
  .addLabel('Tumor_vital', 2)
  .addLabel('Angioinvasion', 2)
  .addLabel('diffuse tumor growth in soft tissue', 2)
  .addLabel('Tumor_necrosis', 2)
  .addLabel('Tumor_regression', 2)
  // Choose server resolution; this should match the resolution at which tiles are exported
  .downsample(downsample)
  // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .multichannelOutput(true) // cucim can't read these image
  .build()
// ----------------------------------------------------------------------------------------------------------------------------------------------------

// --- Write images to pyramidal files ------------------------------------------------------------------------------------------------------
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def annoPath = buildFilePath(annoDir, name + '-qupath-labels.' + format)

new OMEPyramidWriter.Builder(fullMaskServer)
    .downsamples(1, 4, 16, 64, 256)
    //.scaledDownsampling(downsample)
    .tileSize(tileSize)
    .parallelize(32)
    .build()
    .writePyramid(annoPath)
// ----------------------------------------------------------------------------------------------------------------------------------------------------