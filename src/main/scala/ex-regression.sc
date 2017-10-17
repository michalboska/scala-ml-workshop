import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("C:\\Users\\Michal\\Desktop\\SparkSample\\src\\main\\resources\\Clean-Ecommerce.csv")

data.printSchema()

data.show(1)

val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length",
  "Time on App",
  "Time on Website",
  "Length of Membership",
  "Yearly Amount Spent"
)).setOutputCol("features")

val output = assembler.transform(data).select("Yearly Amount Spent", "features")
output.printSchema()
output.show()

val lrModel = new LinearRegression().setLabelCol("Yearly Amount Spent").fit(output)

println(s"Coeff: ${lrModel.coefficients}, intercept: ${lrModel.intercept}")

val summary = lrModel.summary

summary.residuals.show()
println(s"RMSE: ${summary.rootMeanSquaredError}")
println(s"MSE: ${summary.meanSquaredError}")
println(s"R2: ${summary.r2}")


