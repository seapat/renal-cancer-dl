/* groovylint-disable VariableTypeRequired */
import qupath.lib.images.writers.ome.OMEPyramidWriter

def imageData = getCurrentImageData()

// --- PARAMETERS ---------------------------------------------------------------------------------------------------------------------------
double downsample = 1d
int tile_size = 512
def format = "ome.tif"
List annosOfInterest = [
  'Tissue', 
  'Tumor_vital', 
  'diffuse tumor growth in soft tissue', 
  'Angioinvasion', 
  'Tumor_necrosis', 
  'Tumor_regression', ]

// Define output path (relative to project)
def file = imageData.getServer().getMetadata().getName()
def name = GeneralTools.getNameWithoutExtension(file)

String imageLocation = imageData.getServer().getPath()
String path = imageLocation.replace(file, "")

println("Image location: " + imageLocation)
println("Output path: " + path)
println("Image name: " + name)
println("File name: " + file)

// ----------------------------------------------------------------------------------------------------------------------------------------------------

// --- Define binary masks --------------------------------------------------------------------------------------------------------------------

annosOfInterest.each { annotationName ->

  String nameOnFile = annotationName.replaceAll(' ', '_')

  def maskServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
    .addLabel(annotationName, 1)          // Each class requires a name and a number
    .downsample(downsample)               // Choose server resolution; this should match the resolution at which tiles are exported
    .multichannelOutput(false) // cucim does not like >:(
    .build()
  new OMEPyramidWriter.Builder(maskServer)
    .tileSize(tile_size)
    .channelsInterleaved()
    .scaledDownsampling(1d, downsample)
    .downsamples(1, 4, 16, 64, 256)
    .parallelize()
    .losslessCompression()
    .allZSlices() 
    .build()
    .writeSeries(buildFilePath(path, nameOnFile + "-" + annotationName + "ome.tif"))
}

def annotations = getAnnotationObjects()
exportObjectsToGeoJson(annotations, buildFilePath(path, nameOnFile + "-" +".json"))
// ----------------------------------------------------------------------------------------------------------------------------------------------------
