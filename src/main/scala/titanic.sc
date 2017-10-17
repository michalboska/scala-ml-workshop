import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession

Logger.getLogger("org").setLevel(Level.ERROR)


val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("C:\\Users\\Michal\\Desktop\\SparkSample\\src\\main\\resources\\titanic.csv")

data.printSchema()
data.show(5)

val logregdataall = (
  data.select(data("Survived").as("label"),
    $"Pclass",
    $"Name",
    $"Sex",
    $"Age",
    $"SibSp",
    $"Parch",
    $"Ticket",
    $"Fare",
    $"Cabin",
    $"Embarked")
  )

val logregdata = logregdataall.na.drop()

val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")


val assembler = (new VectorAssembler()
  .setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkVec"))
  .setOutputCol("features"))

val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)

val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)

//EVAL
val predictionAndLabels = results.select("prediction", "label").as[(Double, Double)]
predictionAndLabels.show()

val metrics = new MulticlassMetrics(predictionAndLabels.rdd)






