/**
  * Created by mudasser on 09/06/16.
  */

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{SQLContext, DataFrame}

import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

 import org.apache.spark.ml.feature.VectorAssembler
 import org.apache.spark.mllib.linalg.Vectors
 import org.apache.spark.ml.classification.RandomForestClassifier
 import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
 import org.apache.spark.ml.tuning.CrossValidator
 import org.apache.spark.ml.Pipeline
 import org.apache.spark.sql.DataFrame
object TimeSeriesModel {

  def main(args: Array[String]): Unit = {

    //set spark context
    val conf = new SparkConf().setAppName("User ClassificationApp")
                              .setMaster("local")
                              .set("spark.executor.memory","4g")

    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    // Load the data from S3 --> json.gz
    val df = sqlContext.read.json("/mnt/interview_data_yJBC/sample_user_0.json.gz")
                            .withColumn("label", lit(0.0))

    //need to use a ForLoop to get the data and Load into Data Frames
    var dfTS: DataFrame = null

    val dfTS0 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_0.json.gz").withColumn("label", lit(0.0))
    val dfTS1 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_1.json.gz").withColumn("label", lit(1.0))
    val dfTS2 =sqlContext.read.json(s"/mnt/interview_data_yJBC/sample_user_2.json.gz").withColumn("label", lit(2.0))

    dfTS.unionAll(dfTS0)
    dfTS.unionAll(dfTS1)
    dfTS.unionAll(dfTS2)

    // to check the lables
    println(dfTS.select("label").distinct)

//    // To convert DF Column 'User_ID' to 'label'
//    val dfTS_f = dfTS.withColumnRenamed("User_ID","label")
//    dfTS_f.printSchema

//    // UDF To Transform the 'label' from Int to Double
//    val toDouble    = udf[Double, Int](_.toDouble)
//    val train = dfTS_f.withColumn("label", toDouble(output("label")))


    // VectorAssembler: Transformer to combine list of columns(0 to 199) into a single vector column
     val x = (1 to 199).map(_.toString).toArray

     val assembler = new VectorAssembler()
       .setInputCols(x)
       .setOutputCol("features")

     val output = assembler.transform(dfTS)

    // select only label(Double) and features Array[Vector.dense]
     val dataTimeSeries = output.select("label","features").cache()

    println(dataTimeSeries.select("label").distinct.show)

    // Train the model and Cross validate
    val rf = new RandomForestClassifier()

    val pipeline = new Pipeline().setStages(Array(rf))

    val cv = new CrossValidator().setNumFolds(10).setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator)

    val cmModel = cv.fit(output)

  }

}