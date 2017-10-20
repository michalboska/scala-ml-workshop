package sk.mimacom.ml.workshop

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}

// https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names

object LinRegression extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)

  val conf = new SparkConf().setAppName(s"LDAExample").setMaster("local[*]").set("spark.executor.memory", "2g").set("spark.sql.warehouse.dir", "file:///spark-warehouse")
  val spark = SparkSession.builder().config(conf).getOrCreate()
  val sc = spark.sparkContext
  val lda = new LinRegression(sc, spark)

  lda.run()
}

class LinRegression(sc: SparkContext, spark: SparkSession) {

  def run(): Unit = {
    println("[Loading data...]")
    val df = loadDataToDataFrame(spark)
    df.show(false)
    var Array(training, test) = splitTrainingAndTestingData(df)
    val pipeline: Pipeline = getPreparedLinReg

    println("[Training...]")
    val model = pipeline.fit(training)
    println("[Trained]")

    println("[Testing...]")
    val metrics: RegressionMetrics = getMetricsAndTestModel(test, model)
    println("[Tested]")
    printTrainingResults(metrics)

    scanRecordsAndComputePrediction(model)
  }

  def loadDataToDataFrame(spark: SparkSession): DataFrame = {
    import spark.sqlContext.implicits._

    val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("src\\main\\resources\\Housing.csv")

    data.describe().show(false)

    val logregdataall = data.select(data("MEDV").as("label"), $"CRIM", $"ZN", $"INDUS", $"CHAS", $"NOX", $"RM", $"AGE", $"DIS", $"RAD", $"TAX", $"PTRATIO", $"BLACKS", $"LSTAT")
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
        println("crime rate per capita [num];")
        println("zoned residential land [num];")
        println("proportion of non-retail business acres per town [num];")
        println("Charles River dummy variable [1 or 0];")
        println("nitric oxides concentration (parts per 10 million) [num];")
        println("average number of rooms per dwelling [num];")
        println("proportion of owner-occupied units built prior to 1940 [num];")
        println("weighted distances to five Boston employment centres [num];")
        println("index of accessibility to radial highways [num];")
        println("full-value property-tax rate per $10,000 [num];")
        println("pupil-teacher ratio by town [num];")
        println("Proportion of blacks (black race) by town (1000(X - 0.63)^2) [num];")
        println("% lower status of the population [num];")
        print(">")

        val inputs = scala.io.StdIn.readLine().split(";")

        var df = List((inputs(0).toDouble,
          inputs(1).toDouble,
          inputs(2).toDouble,
          inputs(3).toInt,
          inputs(4).toDouble,
          inputs(5).toDouble,
          inputs(6).toDouble,
          inputs(7).toDouble,
          inputs(8).toDouble,
          inputs(9).toDouble,
          inputs(10).toDouble,
          inputs(11).toDouble,
          inputs(12).toDouble
        )).toDF("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "BLACKS", "LSTAT")

        val results2 = model.transform(df)
        results2.select("prediction").show(false)

        println("Exit:")
        exitInput = String.valueOf(scala.io.StdIn.readLine())
      }
      catch {
        case e: Exception => println("Bad input!" + e.toString)
      }

    } while (!exitInput.equals("y"))
  }

  private def printTrainingResults(metrics: RegressionMetrics): Unit = {
    println("Regression metrics:")
    println("Mean abs error:" + metrics.meanAbsoluteError)
    println("Mean sq error:" + metrics.meanSquaredError)
    println("RMSE:" + metrics.rootMeanSquaredError)
  }

  private def getMetricsAndTestModel(test: Dataset[Row], model: PipelineModel) = {
    import spark.sqlContext.implicits._
    val results = model.transform(test)
    val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
    val metrics = new RegressionMetrics(predictionAndLabels)
    metrics
  }

  private def getPreparedLinReg = {


    val assemblerFeatures = new VectorAssembler()
      .setInputCols(Array("CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "BLACKS", "LSTAT"))
      .setOutputCol("rawFeatures")

    val scaler = new StandardScaler().setInputCol("rawFeatures").setOutputCol("features")
    val lr = new LinearRegression()
      .setMaxIter(100)

    val pipeline = new Pipeline().setStages(
      Array(assemblerFeatures, scaler, lr)
    )
    pipeline
  }
}

