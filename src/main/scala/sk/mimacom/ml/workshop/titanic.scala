package sk.mimacom.ml.workshop

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}


object Titanic extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  val conf = new SparkConf().setAppName(s"LDAExample").setMaster("local[*]").set("spark.executor.memory", "2g").set("spark.sql.warehouse.dir", "file:///spark-warehouse")
  val spark = SparkSession.builder().config(conf).getOrCreate()
  val sc = spark.sparkContext
  val lda = new Titanic(sc, spark)

  lda.run()
}

class Titanic(sc: SparkContext, spark: SparkSession) {

  def run(): Unit = {
    println("[Loading data...]")
    val df = loadDataToDataFrame(spark)
    var Array(training, test) = splitTrainingAndTestingData(df)
    val pipeline: Pipeline = getPrepearedLogisticReg

    println("[Training...]")
    val model = pipeline.fit(training)
    println("[Trained]")

    println("[Testing...]")
    val metrics: MulticlassMetrics = getMetricsAndTestModel(test, model)
    println("[Tested]")
    printTrainingResults(metrics)

    scanRecordsAndComputePrediction(model)
  }

  def loadDataToDataFrame(spark: SparkSession): DataFrame = {
    import spark.sqlContext.implicits._

    val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src\\main\\resources\\titanic.csv")

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

    logregdata
  }

  def splitTrainingAndTestingData(df: DataFrame): Array[Dataset[Row]] = {

    var Array(training, test) = df.randomSplit(Array(70, 30), seed = 12345)
    Array(training, test)
  }

  def scanRecordsAndComputePrediction(model: PipelineModel): Any = {
    import spark.sqlContext.implicits._

    var exitInput = "n"

    do {

      try {
        println("TYPE ONE LINE INPUT.\n SYNTAX:")
        println("Pclass         [num]{1,2,3};")
        println("Sex            [male | female];")
        println("Age            [num];")
        println("Num of siblings/spouses    [num];")
        println("Num of parents/children    [num];")
        println("Price paid for ticket      [num];")
        println("City where boarded         [C(herbourg) | Q(ueenstown) | S(outhampton)];")
        print(">")

        val inputs = scala.io.StdIn.readLine().split(";")

        var df = List((inputs(0).toInt,
          inputs(1),
          inputs(2).toInt,
          inputs(3).toInt,
          inputs(4).toInt,
          inputs(5).toDouble,
          inputs(6))).toDF("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")


        val results2 = model.transform(df)
        results2.select("probability", "prediction").show(false)

        println("Exit:")
        exitInput = String.valueOf(scala.io.StdIn.readLine())
      }
      catch {
        case e: Exception => println("Bad input!" + e.toString)
      }

    } while (!exitInput.equals("y"))
  }

  private def printTrainingResults(metrics: MulticlassMetrics): Unit = {
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    println("PRECISION OF PREDICTION: " + metrics.accuracy)
  }

  private def getMetricsAndTestModel(test: Dataset[Row], model: PipelineModel) = {
    import spark.sqlContext.implicits._
    val results = model.transform(test)
    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
    metrics
  }

  private def getPrepearedLogisticReg = {
    val sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

    val sexEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
    val embarkedEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVec")

    val assemblerFeatures = new VectorAssembler()
      .setInputCols(Array("Pclass", "SexVec", "Age", "SibSp", "Parch", "Fare", "EmbarkedVec"))
      .setOutputCol("features")

    val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(
      Array(sexIndexer, embarkedIndexer, sexEncoder, embarkedEncoder, assemblerFeatures, lr)
    )
    pipeline
  }
}