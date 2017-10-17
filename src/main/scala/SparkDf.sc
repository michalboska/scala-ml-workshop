import org.apache.spark.sql.SparkSession

val session = SparkSession.builder().getOrCreate()

val df = session.read.option("header", "true").option("inferSchema", "false").csv("DOCUMENTMANAGER_FINANCIAL_DOCUMENT.csv")

for (row <- df.head(10)) {
  println(row)
}

val docNrColumn = df.select("DOCUMENT_NUMBER")