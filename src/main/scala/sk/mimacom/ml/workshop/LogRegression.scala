package sk.mimacom.ml.workshop

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}


object LogRegression extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  val conf = new SparkConf().setAppName(s"LDAExample").setMaster("local[*]").set("spark.executor.memory", "2g").set("spark.sql.warehouse.dir", "file:///spark-warehouse")
  val spark = SparkSession.builder().config(conf).getOrCreate()
  val sc = spark.sparkContext
  val lda = new LogRegression(sc, spark)

  lda.run()
}

class LogRegression(sc: SparkContext, spark: SparkSession) {

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

    val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src\\main\\resources\\bank-full.csv")

    val logregdataall = data.select(data("y").as("label"), $"age", $"job", $"marital", $"education", $"default", $"balance", $"housing", $"loan", $"contact", $"day", $"month", $"duration", $"campaign", $"pdays", $"previous", $"poutcome")

    val logregdata = logregdataall.na.drop()

    logregdata
  }

  def splitTrainingAndTestingData(df: DataFrame): Array[Dataset[Row]] = {

    var Array(training, test) = df.randomSplit(Array(70, 30), seed = 12345)

    //    training = training.select("*").where("label = 1").limit(1000).union(training.select("*").where("label = 0").limit(1000)).orderBy(rand())
    //    test = test.select("*").where("label = 1").limit(2000).union(test.select("*").where("label = 0").limit(2000)).orderBy(rand())

    Array(training, test)
  }

  def scanRecordsAndComputePrediction(model: PipelineModel): Any = {
    import spark.sqlContext.implicits._
    var exitInput = "n"

    do {

      try {
        println("TYPE ONE LINE INPUT.\n SYNTAX:")
        println("age        [num];")
        println("job        [admin.| management | technician | blue-collar | entrepreneur | housemaid | services | retired | student | unemployed];")
        println("education  [primary | secondary | tertiary ];")
        println("marital    [single | married | divorced]")
        print(">")

        val inputs = scala.io.StdIn.readLine().split(";")

        var df = List((inputs(0).toInt, inputs(1), inputs(2), inputs(3))).toDF("age", "job", "education", "marital")

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
    val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("JobIndex")
    val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("MaritalIndex")
    val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("EducationIndex")
    val contactIndexer = new StringIndexer().setInputCol("contact").setOutputCol("ContactIndex")
    val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("MonthIndex")
    val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("PoutcomeIndex")


    val jobEncoder = new OneHotEncoder().setInputCol("JobIndex").setOutputCol("JobVec")
    val maritalEncoder = new OneHotEncoder().setInputCol("MaritalIndex").setOutputCol("MaritalVec")
    val educationEncoder = new OneHotEncoder().setInputCol("EducationIndex").setOutputCol("EducationVec")
    val contactEncoder = new OneHotEncoder().setInputCol("ContactIndex").setOutputCol("ContactVec")
    val monthEncoder = new OneHotEncoder().setInputCol("MonthIndex").setOutputCol("MonthVec")
    val poutcomeEncoder = new OneHotEncoder().setInputCol("PoutcomeIndex").setOutputCol("PoutcomeVec")


    val assemblerFeatures = new VectorAssembler()
      .setInputCols(Array("JobVec", "MaritalVec", "EducationVec", "ContactVec", "MonthVec", "PoutcomeVec", "age", "default", "balance", "housing", "loan", "day", "duration", "campaign", "pdays", "previous"))
      .setOutputCol("features")

    val lr = new LogisticRegression()

    val pipeline = new Pipeline().setStages(
      Array(jobIndexer, maritalIndexer, educationIndexer, contactIndexer, monthIndexer, poutcomeIndexer,
        jobEncoder, maritalEncoder, educationEncoder, contactEncoder, monthEncoder, poutcomeEncoder, assemblerFeatures, lr)
    )
    pipeline
  }
}

