import warnings
import findspark
import pandas as pd
import seaborn as sns
from pyspark.ml.classification import GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, lower
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as F
from pyspark.sql.functions import when, count, col
from pyspark.sql.functions import col, explode, array, lit


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

findspark.init(r"C:\Users\HP\spark-3.2.0-bin-hadoop2.7\spark-3.2.0-bin-hadoop2.7")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_giris") \
    .getOrCreate()
sc = spark.sparkContext
#sc.stop()

spark_df = spark.read.csv(r"C:\Users\HP\Desktop\CHURN-PREDICTION\source\churn2.csv", header=True, inferSchema=True)
############################
# Exploratory Data Analysis
############################
print("Shape: ", (spark_df.count(), len(spark_df.columns)))
spark_df.show()
spark_df.dtypes
spark_df = spark_df.toDF(*[c.upper() for c in spark_df.columns])
spark_df.describe().toPandas().transpose()
spark_df.select("EXITED").distinct().show()


num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
num_cols.remove('ROWNUMBER')
num_cols.remove('CUSTOMERID')
spark_df.select(num_cols).describe().toPandas().transpose()

# Tüm kategorik değişkenlerin seçimi ve özeti
cat_cols = [col[0] for col in spark_df.dtypes if col[1] == 'string']
cat_cols.remove('SURNAME')

for col in cat_cols:
    spark_df.select(col).distinct().show()

for col in [col.lower() for col in num_cols]:
    spark_df.groupby("EXITED").agg({col: "mean"}).show()

############################
# Missing Values
############################
#spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T
spark_df.dropna().show()
spark_df.count()

############################
# Label Encoding
############################
indexer = StringIndexer(inputCol="GENDER", outputCol="GENDER_LABEL")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("GENDER_LABEL", temp_sdf["GENDER_LABEL"].cast("integer"))
spark_df.show(5)
spark_df = spark_df.drop('GENDER')


indexer = StringIndexer(inputCol="GEOGRAPHY", outputCol="GEOGRAPHY_LABEL")
indexer.fit(spark_df).transform(spark_df).show(5)
temp_sdf = indexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("GEOGRAPHY_LABEL", temp_sdf["GEOGRAPHY_LABEL"].cast("integer"))
spark_df.show(5)
spark_df = spark_df.drop('GEOGRAPHY')
############################
# Feature Interaction
############################
spark_df = spark_df.withColumn('AGE_CREDITSCORE_RATIO', spark_df.AGE / spark_df.CREDITSCORE*100)
spark_df.show(5)
############################
# Bucketization / Bining / Num to Cat
############################
#bucketizer = Bucketizer(splits=[0, 35, 45, 65], inputCol="AGE", outputCol="AGE_CAT")
#spark_df = bucketizer.setHandleInvalid("keep").transform(spark_df)
#spark_df = spark_df.withColumn('AGE_CAT', spark_df.AGE_CAT + 1)
#spark_df = spark_df.withColumn("AGE_CAT", spark_df["AGE_CAT"].cast("integer"))
#spark_df.show()

############################
# One Hot Encoding
############################
encoder = OneHotEncoder(inputCols=["GEOGRAPHY_LABEL"], outputCols=["GEOGRAPHY_LABEL_OHE"])
spark_df = encoder.fit(spark_df).transform(spark_df)
spark_df.show(5)

############################
# TARGET'ın Tanımlanması
############################
stringIndexer = StringIndexer(inputCol='EXITED', outputCol='label')
temp_sdf = stringIndexer.fit(spark_df).transform(spark_df)
spark_df = temp_sdf.withColumn("label", temp_sdf["label"].cast("integer"))
spark_df.show(5)

############################
# Feature'ların Tanımlanması
############################
num_cols = [col[0] for col in spark_df.dtypes if col[1] != 'string']
num_cols.remove('ROWNUMBER')
num_cols.remove('CUSTOMERID')
num_cols.remove('label')
num_cols.remove('EXITED')
spark_df.select(num_cols).show()

# Vectorize independent variables.
va = VectorAssembler(inputCols=num_cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show()

# Final sdf
final_df = va_df.select("features", "label")
final_df.show(5)


train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=17)
train_df.show(10)
test_df.show(10)

#########################BALANCE############################
train_df.groupby("label").count().show()
major_df = train_df.filter(F.col("label") == 0)
minor_df = train_df.filter(F.col("label") == 1)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))

sampled_majority_df = major_df.sample(False, 1/ratio)
combined_df_2 = sampled_majority_df.unionAll(minor_df)
train_df=combined_df_2

print("Training Dataset Count: " + str(train_df.count()))
print("Test Dataset Count: " + str(test_df.count()))
train_df.groupby("label").count().show()
##################################################
# Modeling
##################################################
############################
# Logistic Regression
############################
log_model = LogisticRegression(featuresCol='features', labelCol='label').fit(train_df)
y_pred = log_model.transform(test_df)
y_pred.show()
y_pred.select("label", "prediction").show()

# accuracy
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction", metricName='areaUnderROC')
evaluatorMulti = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
acc = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "accuracy"})
precision = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "precisionByLabel"})
recall = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "recallByLabel"})
f1 = evaluatorMulti.evaluate(y_pred, {evaluatorMulti.metricName: "f1"})
roc_auc = evaluator.evaluate(y_pred)
print("accuracy: %f, precision: %f, recall: %f, f1: %f, roc_auc: %f" % (acc, precision, recall, f1, roc_auc))
############################
# Gradient Boosted Tree Classifier
############################
gbm = GBTClassifier(maxIter=100, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show(5)
y_pred.filter(y_pred.label == y_pred.prediction).count() / y_pred.count()
############################
# Model Tuning
############################
evaluator = BinaryClassificationEvaluator()
gbm_params = (ParamGridBuilder()
              .addGrid(gbm.maxDepth, [2, 4, 6])
              .addGrid(gbm.maxBins, [20, 30])
              .addGrid(gbm.maxIter, [10, 20])
              .build())
cv = CrossValidator(estimator=gbm,estimatorParamMaps=gbm_params,evaluator=evaluator,numFolds=5)
cv_model = cv.fit(train_df)
y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()