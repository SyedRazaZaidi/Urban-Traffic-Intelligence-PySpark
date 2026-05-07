import os
import csv
from flask import Flask, jsonify, request, render_template
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, round as spark_round, hour, to_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np

# --- WINDOWS HADOOP FIX ---
os.environ['HADOOP_HOME'] = "C:\\hadoop"
os.environ['SPARK_LOCAL_DIRS'] = "C:\\hadoop\\tmp"

app = Flask(__name__)

# ==========================================
# 1. SPARK INITIALIZATION & DATA INGESTION
# ==========================================
spark = SparkSession.builder \
    .appName("UrbanTrafficIntelligence") \
    .master("local[*]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.sql.warehouse.dir", "file:///C:/hadoop/tmp") \
    .getOrCreate()

dataset_path = "traffic_dataset.csv"

# FAIL-SAFE: Generate a structurally identical dataset if the real one isn't downloaded yet
if not os.path.exists(dataset_path):
    print(f"Real dataset missing. Auto-generating fallback dataset at: {dataset_path}...")
    np.random.seed(42)
    with open(dataset_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["temp", "clouds_all", "weather_main", "date_time", "traffic_volume"])
        weather_types = ["Clear", "Clouds", "Rain", "Snow", "Mist"]
        for i in range(5000):
            hr = np.random.randint(0, 24)
            temp = np.random.uniform(250, 310)
            clouds = np.random.randint(0, 100)
            weather = np.random.choice(weather_types)
            vol = int((100 if hr in [8,9,17,18] else 20) * 30 + temp - clouds + np.random.normal(0, 500))
            writer.writerow([round(temp, 2), clouds, weather, f"2012-10-02 {hr:02d}:00:00", max(100, vol)])

print("Loading CSV into Spark DataFrame...")
raw_df = spark.read.csv(dataset_path, header=True, inferSchema=True)

# Clean and transform: Extract the specific 'Hour' from the timestamp string
df = raw_df.withColumn("HourOfDay", hour(to_timestamp(col("date_time")))) \
           .select("temp", "clouds_all", "HourOfDay", "weather_main", "traffic_volume") \
           .dropna().cache()

# ==========================================
# 2. MLLIB DISTRIBUTED PREDICTOR
# ==========================================
assembler = VectorAssembler(inputCols=["temp", "clouds_all", "HourOfDay"], outputCol="features")
lr_model = None

def train_model():
    global lr_model
    assembled_df = assembler.transform(df)
    train_data, test_data = assembled_df.randomSplit([0.8, 0.2], seed=42)
    
    lr = LinearRegression(featuresCol="features", labelCol="traffic_volume")
    lr_model = lr.fit(train_data)
    
    predictions = lr_model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="traffic_volume", predictionCol="prediction", metricName="rmse")
    return evaluator.evaluate(predictions)

print(f"Booting Spark Engine... Initial MLlib RMSE: {train_model():.2f} cars")

# ==========================================
# 3. FLASK API ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Spark SQL Endpoint: Group big data by weather condition to find average traffic."""
    try:
        analytics_df = df.groupBy("weather_main").agg(
            spark_round(avg("traffic_volume"), 0).alias("AvgTrafficVolume"),
            spark_round(avg("temp"), 2).alias("AvgTemperature"),
            spark_round(avg("clouds_all"), 1).alias("AvgCloudCover")
        ).orderBy("AvgTrafficVolume", ascending=False)
        
        results = analytics_df.toPandas().to_dict(orient="records")
        return jsonify({"status": "success", "data": results})
    except Exception as e:
         return jsonify({"status": "error", "message": str(e)})

@app.route('/api/train', methods=['POST'])
def trigger_training():
    """MLlib Endpoint: Retrain the cluster weights on demand."""
    try:
        rmse = train_model()
        return jsonify({"status": "success", "rmse": round(rmse, 2)})
    except Exception as e:
         return jsonify({"status": "error", "message": str(e)})

@app.route('/api/predict', methods=['POST'])
def predict_volume():
    """Integration Endpoint: Test a single row of data against the trained model."""
    if lr_model is None:
        return jsonify({"status": "error", "message": "Model not trained."})
    
    data = request.json
    try:
        input_df = spark.createDataFrame([
            (float(data['temp']), float(data['clouds']), int(data['hour']))
        ], ["temp", "clouds_all", "HourOfDay"])
        
        prediction_df = lr_model.transform(assembler.transform(input_df))
        predicted_cars = max(0, int(prediction_df.select("prediction").collect()[0][0]))
        
        return jsonify({"status": "success", "predicted_volume": predicted_cars})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)