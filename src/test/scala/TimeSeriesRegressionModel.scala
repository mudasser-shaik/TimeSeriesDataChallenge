/**
  * Created by mudasser on 09/06/16.
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

object TimeSeriesRegressionModel {

  def main(args: Array[String]): Unit = {

    //set spark context
    val conf = new SparkConf().setAppName("User ClassificationApp")
                              .setMaster("local")
                              .set("spark.executor.memory","4g")

    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // Load the data from S3 --> json.gz
    val df = sqlContext.read.json("/mnt/dataChallengeTS/sample_user_0.json.gz")
                            .withColumn("label", lit(0.0))

    //Need to use a ForLoop to get the data from dir and Load into Data Frames
  /*
    var dfTS: DataFrame = null;
    for (file <- files) {
    val fileDf= sc.textFile(file)
      if (df!= null) {
         df= df.unionAll(fileDf)
       } else {
      df= fileDf
      }
     }
  */

    var dfTS: DataFrame = null

    val dfTS0 =sqlContext.read.json(s"/mnt/dataChallengeTS/sample_user_0.json.gz").withColumn("label", lit(0.0))
    val dfTS1 =sqlContext.read.json(s"/mnt/dataChallengeTS/sample_user_1.json.gz").withColumn("label", lit(1.0))
    val dfTS2 =sqlContext.read.json(s"/mnt/dataChallengeTS/sample_user_2.json.gz").withColumn("label", lit(2.0))

    dfTS.unionAll(dfTS0)
    dfTS.unionAll(dfTS1)
    dfTS.unionAll(dfTS2)

    // Check the distinct lables
    println(dfTS.select("label").distinct)

    // VectorAssembler: Transformer to combine list of columns(0 to 199) into a single vector column
     val x = (1 to 199).map(_.toString).toArray

     val assembler = new VectorAssembler()
       .setInputCols(x)
       .setOutputCol("features")

     val output = assembler.transform(dfTS)

    // select only label(Double) and features Array[Vector.dense] and Cache
     val dataTimeSeries = output.select("label","features").cache()

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = dataTimeSeries.randomSplit(Array(0.7, 0.3))

     // Train a RandomForest model.
     val rf = new RandomForestRegressor()
       .setLabelCol("label")
       .setFeaturesCol("features")

     // Chain forest in a Pipeline.
     val pipeline = new Pipeline().setStages(Array(rf))

     // Train model.
     val model = pipeline.fit(trainingData)

     // Make predictions.
     val predictions = model.transform(testData)

     println(predictions.select("prediction", "label", "features").show(5))

    // Select (prediction, true label) and compute test error
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val rfModel = model.stages(0).asInstanceOf[RandomForestRegressionModel]
    println("Learned regression forest model:\n" + rfModel.toDebugString)
  }

}
