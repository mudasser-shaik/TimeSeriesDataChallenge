# TimeSeries
Data challenge on TimeSeries data. 
Each User has a File sample_user_(0 to 9)

val timeSeriesDF = sqlContext.read.json("/mnt/interview_data_yJBC/sample_user_0.json.gz")
timeSeriesDF.printSchema()

User*
|--root 
    |-- 0: double (nullable = true)
    |-- 1: double (nullable = true) 
    |-- 10: double (nullable = true)
    |-- 100: double (nullable = true) 
    |-- 101: double (nullable = true) 
    |-- 102: double (nullable = true) 
    |-- 103: double (nullable = true) 
    |-- 104: double (nullable = true) 
    |-- 105: double (nullable = true) 
    |-- 106: double (nullable = true)
    |-- 107: double (nullable = true) 
    |-- 108: double (nullable = true)
    ......
    |-- 199: double (nullable = true
    

Label    -> User-> 0 to 9
features -> values -> 0 to 199
