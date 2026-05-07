# Urban-Traffic-Intelligence-PySpark
An end-to-end Big Data pipeline utilizing Apache Spark SQL, MLlib, and Flask to predict urban traffic volume from real-world environmental sensor data.
# 🌐 NEXUS: AI-Based Urban Traffic Intelligence System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey.svg)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?logo=tailwind-css&logoColor=white)

An end-to-end distributed Big Data web application built to process, aggregate, and learn from massive urban traffic datasets. This project serves as a Minimum Viable Product (MVP) for smart city infrastructure, demonstrating the integration of a backend data processing engine with a modern, reactive web interface.

## 🚀 Project Overview

Modern smart cities generate millions of IoT sensor logs daily. Handling this requires robust big data architectures. **NEXUS** ingests over 48,000 real-world traffic records (Metro Interstate Traffic Volume Dataset) to extract high-level telemetry and predict localized traffic congestion based on environmental metrics.

### Key Features
* **Big Data Processing (Spark SQL):** Aggregates massive datasets by environmental conditions (weather, temperature, cloud cover) to extract actionable telemetry in real-time.
* **Distributed Machine Learning (Spark MLlib):** Utilizes a `VectorAssembler` and distributed `LinearRegression` pipeline to predict continuous traffic volume (cars per hour) based on temporal and weather inputs.
* **RESTful Middleware (Flask):** A lightweight API layer that translates asynchronous frontend HTTP requests into distributed PySpark background jobs.
* **Premium UI/UX:** A responsive, dark-mode Glassmorphism dashboard featuring animated `Chart.js` data visualization and Tailwind CSS styling.

---

## 📸 System Dashboard

*(Add your screenshots here by dragging and dropping them into the GitHub editor)*

| Live Congestion Predictor | Spark SQL Telemetry & Chart.js |
| :---: | :---: |
| `![UI Screenshot 1](link-to-image)` | `![UI Screenshot 2](link-to-image)` |

---

## ⚙️ System Architecture

1. **The Backend Engine:** Apache PySpark manages ETL workloads, bypassing standard local memory constraints to train MLlib weights on physical CSV datasets.
2. **The API Layer:** Flask exposes three core endpoints:
   * `GET /api/analytics`: Executes distributed SQL grouping.
   * `POST /api/train`: Triggers on-demand model weight syncing.
   * `POST /api/predict`: Runs real-time inference on live user input.
3. **The Frontend:** Vanilla JavaScript asynchronously fetches Spark data to dynamically render charts and UI states without page reloads.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* Java 8 or 11 (Required for Apache Spark JVM)

### Running Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/SyedRazaZaidi/Urban-Traffic-Intelligence-PySpark.git](https://github.com/SyedRazaZaidi/Urban-Traffic-Intelligence-PySpark.git)
   cd Urban-Traffic-Intelligence-PySpark
