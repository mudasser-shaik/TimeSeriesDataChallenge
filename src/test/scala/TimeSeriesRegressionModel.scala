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
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

object TimeSeriesClassificationModel {

  def main(args: Array[String]): Unit = {

    //set spark context
    val conf = new SparkConf().setAppName("User ClassificationApp")
      .setMaster("local")
      .set("spark.executor.memory","4g")

    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //Need to use a ForLoop to get the data from dir and Load into Data Frames --> sc.WholeText(....)
    val files = (0 to 9).map(i => s"/mnt/interview_data_yJBC/sample_user_$i.json.gz").toArray

    // Load the data from Dir json.gz (can be done in a better way)
    var dfTS: DataFrame = null

    /*
        var i =0.0
        for (file <- files) {
          val fileDf = sqlContext.read.json(file).withColumn("label", lit(i))
          i +=1
          if (dfTS!= null) {
             dfTS= dfTS.unionAll(fileDf)
           } else {
          dfTS= fileDf
          }
         }
    */

    val dfTS0 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_0.json.gz").withColumn("label", lit(0.0))
    val dfTS1 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_1.json.gz").withColumn("label", lit(1.0))
    val dfTS2 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_2.json.gz").withColumn("label", lit(2.0))
    val dfTS3 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_3.json.gz").withColumn("label", lit(3.0))
    val dfTS4 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_4.json.gz").withColumn("label", lit(4.0))
    val dfTS5 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_5.json.gz").withColumn("label", lit(5.0))
    val dfTS6 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_6.json.gz").withColumn("label", lit(6.0))
    val dfTS7 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_7.json.gz").withColumn("label", lit(7.0))
    val dfTS8 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_8.json.gz").withColumn("label", lit(8.0))
    val dfTS9 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_9.json.gz").withColumn("label", lit(9.0))

    dfTS.unionAll(dfTS0)
    dfTS.unionAll(dfTS1)
    dfTS.unionAll(dfTS2)

    // Check the distinct lables
    println(dfTS.select("label").distinct)

    // VectorAssembler: Transformer to combine list of columns(0 to 199) into a single vector column
    val xCol = (1 to 199).map(_.toString).toArray

    val assembler = new VectorAssembler()
          .setInputCols(xCol)
          .setOutputCol("features")

    val output = assembler.transform(dfTS)

    // select only label(Double) and features Array[Vector] and Cache
    val dataTimeSeries = output.select("label","features").cache()

    // Saving it into parquet File and registering as Tabel and Querying it
    dataTimeSeries.write.parquet("UserTimeSeries.parquet")
    val userdata = sqlContext.read.parquet("UserTimeSeries.parquet")
    userdata.registerTempTable("UserTimeSeriesTable")

    //select the required features for classification
    sqlContext.sql("SELECT label from UserTimeSeriesTable WHERE label > 8").show

    // ----------------------Random Forest Classification Model-------------------

    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
    import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(dataTimeSeries)

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(dataTimeSeries)


    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = dataTimeSeries.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Train model.  This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    predictions.select("predictedLabel").distinct.show()

    // Select (prediction, true label) and compute test error
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")

    val accuracy = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))

  }

}
