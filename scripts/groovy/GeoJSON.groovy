
setDefaultImageData(null)
String imageData = getCurrentImageData()

println imageData
println fileExists(imageData)
println "\n"

if (imageData){
    // We do not have some svs files, those are not usable becausewe lack metadata anyways
    // but they are still in the qupatch project

    String imageLocation = getCurrentImageData().getServer().getPath()
    println imageLocation
    println fileExists(imageLocation)
    println "\n"

    def annotations = getAnnotationObjects()



    String path = imageLocation.split("file:")[1]//imageLocation.split("file:/")[1]
    println path
    println fileExists(path)
    println "\n"

    file = path.split(".svs")[0] //+ ".json" //GeneralTools.getNameWithoutExtension(path+".json")
    println "\n"
    println file
    println "\n"

    // The same method without the 'FEATURE_COLLECTION' parameter outputs a simple JSON object/array

    exportObjectsToGeoJson(annotations, file +".json")
    }


// 'FEATURE_COLLECTION' is standard GeoJSON format for multiple objects
// exportObjectsToGeoJson(annotations, path+"-FeatColl.json", "FEATURE_COLLECTION")