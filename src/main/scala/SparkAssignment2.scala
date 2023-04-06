import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object SparkAssignment2 extends App {

  // creating spark session
  implicit val spark: SparkSession = SparkSession
    .builder()
    .appName("SparkAssignment2")
    .master("local[*]")
    .getOrCreate()

  spark.sparkContext.setLogLevel("ERROR") // We want to ignore all of the INFO and WARN messages.

  // Read train data from CSV file
  val train = try {
    spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("src/main/resources/train.csv")
  } catch {
    case e: Exception =>
      println(s"An error occurred while reading the ratings CSV file: ${e.getMessage}")
      spark.emptyDataFrame
  }

  // Read test data from CSV file
  val test = try {
    spark.read
      .option("header", true)
      .option("inferSchema", true)
      .csv("src/main/resources/test.csv")
  } catch {
    case e: Exception =>
      println(s"An error occurred while reading the ratings CSV file: ${e.getMessage}")
      spark.emptyDataFrame
  }

  /// ********** SECTION 1: EXPLORATORY DATA ANALYSIS ********** ///

  // Show the first 5 rows of the dataset
  train.show(5)

  // Print the schema of the dataset
  train.printSchema()

  // Compute summary statistics for numeric columns
  train.describe("Age", "SibSp", "Parch", "Fare").show()

  // Compute the count of each categorical value in the Sex column
  train.groupBy("Sex").count().show()

  // Compute the count of each categorical value in the Pclass column
  train.groupBy("Pclass").count().show()

  // Compute the count of each categorical value in the Embarked column
  train.groupBy("Embarked").count().show()

  // Compute the mean age of passengers by sex and class
  train.groupBy("Sex", "Pclass")
    .agg(avg("Age").alias("mean_age"))
    .show()

  // Compute the survival rate by sex and class
  train.groupBy("Sex", "Pclass")
    .agg((sum(when(col("Survived") === 1, 1).otherwise(0)) / count("*")).alias("survival_rate"))
    .show()

  /// ********** SECTION 2: FEATURE ENGINEERING ********** ///

  // Creating attribute FamSize - total family size = SibSp + Parch + 1
  val familyUDF = udf((SibSp: Int, Parch: Int) => SibSp +Parch + 1)
  val train_v2 = train.withColumn("FamSize", familyUDF(col("SibSp"), col("Parch")))
  val test_v2 = test.withColumn("FamSize", familyUDF(col("SibSp"), col("Parch")))

  // Dropping attributes SibSp and Parch
  val train_v2_1 = train_v2.drop("SibSp", "Parch")
  val test_v2_1 = test_v2.drop("SibSp", "Parch")

  // Creating attribute FarePerPerson - Fare per person = Fare / FamSize
  val train_v3 = train_v2_1.withColumn("FarePerPerson", col("Fare") / col("FamSize"))
  val test_v3 = test_v2_1.withColumn("FarePerPerson", col("Fare") / col("FamSize"))

  // Creating attribute Age group
  // Define the age groups
  val ageGroups = Map(
    0 -> "Unknown",
    1 -> "Infant",
    2 -> "Toddler",
    3 -> "Child",
    4 -> "Teenager",
    5 -> "Young Adult",
    6 -> "Adult",
    7 -> "Senior"
  )

  val train_final = train_v3.withColumn(
    "AgeGroup",
    when(col("Age").isNull, ageGroups(0))
      .when(col("Age") <= 1, ageGroups(1))
      .when(col("Age") <= 3, ageGroups(2))
      .when(col("Age") <= 12, ageGroups(3))
      .when(col("Age") <= 19, ageGroups(4))
      .when(col("Age") <= 35, ageGroups(5))
      .when(col("Age") <= 65, ageGroups(6))
      .otherwise(ageGroups(7))
  )

  val test_final = test_v3.withColumn(
    "AgeGroup",
    when(col("Age").isNull, ageGroups(0))
      .when(col("Age") <= 1, ageGroups(1))
      .when(col("Age") <= 3, ageGroups(2))
      .when(col("Age") <= 12, ageGroups(3))
      .when(col("Age") <= 19, ageGroups(4))
      .when(col("Age") <= 35, ageGroups(5))
      .when(col("Age") <= 65, ageGroups(6))
      .otherwise(ageGroups(7))
  )

//  train_final.printSchema()
//  train_final.show(10)
//
//  test_final.printSchema()
//  test_final.show(10)

  /// ********** SECTION 3: PREDICTION ********** ///

  // Define the features to use in the model
  val features = Array("Pclass", "Sex", "Fare",
    "Embarked", "FamSize", "FarePerPerson", "AgeGroup")

  // Create StringIndexers for the Sex and Embarked columns
  val sexIndexer = new StringIndexer()
    .setInputCol("Sex")
    .setOutputCol("SexIndex")
    .setHandleInvalid("skip")

  val embarkedIndexer = new StringIndexer()
    .setInputCol("Embarked")
    .setOutputCol("EmbarkedIndex")
    .setHandleInvalid("skip")

  val ageGroupIndexer = new StringIndexer()
    .setInputCol("AgeGroup")
    .setOutputCol("AgeGroupIndex")
    .setHandleInvalid("skip")

  // Create a VectorAssembler to combine the features into a single vector
  val assembler = new VectorAssembler()
    .setInputCols(Array("Fare", "Pclass", "FamSize", "FarePerPerson"))
    .setOutputCol("features")

  // Create a RandomForest model
  val rf = new RandomForestClassifier()
    .setLabelCol("Survived")
    .setFeaturesCol("features")

  // Create a Pipeline to string together the stages
  val pipeline = new Pipeline()
    .setStages(Array(sexIndexer, embarkedIndexer, ageGroupIndexer, assembler, rf))

  // Split the data into training and validation sets
  val Array(trainingData, validationData) = train_final.randomSplit(Array(0.8, 0.2))

  // Fit the model to the training data
  val model = pipeline.fit(trainingData)

  // Make predictions on the validation data
  val predictions = model.transform(validationData)

  // Evaluate the model using the area under the ROC curve
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("Survived")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  val accuracy = evaluator.evaluate(predictions)

  // Print the accuracy
  println(s"Accuracy: $accuracy")

  // Use the trained model to make predictions on the test data
  val predictions_test = model.transform(test_final)

  // Show the predicted survival probabilities for each passenger
  predictions_test.select("PassengerId", "prediction", "probability").show()

}
