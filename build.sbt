name := "TrueMotionTimeSeries"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.apache.spark" %% "spark-sql" % "1.6.0",
  "org.apache.spark" %% "spark-mllib" % "1.6.0",
  "com.databricks" %% "spark-csv" % "1.4.0",
  "org.apache.spark" %% "spark-hive" % "1.6.0",
  "org.apache.spark" %% "spark-streaming" % "1.6.0",
  "org.apache.spark" %% "spark-streaming-kafka" % "1.6.0",
  "com.github.scopt" % "scopt_2.11" % "3.4.0",
  "org.specs2" %% "specs2-core" % "3.7" % "test",
  "org.specs2" %% "specs2-junit" % "3.7" % "test",
  "org.specs2" % "specs2_2.11" % "3.7"
)