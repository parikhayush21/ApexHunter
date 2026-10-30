# 🏎️ ApexHunter 2.0 — F1 Telemetry & Onboard-Video Analytics

[![ApexHunter CI/CD](https://github.com/parikhayush21/ApexHunter/actions/workflows/ci-cd.yml/badge.svg?branch=main)](https://github.com/parikhayush21/ApexHunter/actions/workflows/ci-cd.yml)

ApexHunter 2.0 analyses Formula 1 driving performance using official telemetry (FastF1, 2023–2024
seasons, ~57.9M rows) and onboard pole-lap video (46 laps). It combines classical machine learning,
deep learning and graph search, and shows the results in an interactive Streamlit dashboard.

**Courses (AY 2026–27):** Data Science & Machine Learning · Deep Learning
*(The project was first built in the previous semester for Artificial Intelligence and Big Data.)*

| Component | Technique | Course |
|---|---|---|
| Driver mistake detection | Isolation Forest + engineered telemetry features, K-fold contamination search | DSML |
| Racing-line search | A\*, Dijkstra, BFS on a telemetry-weighted grid | DSML |
| Tyre cliff prediction | 2-layer LSTM, grid search, Monte Carlo dropout uncertainty | Deep Learning |
| Apex detection | YOLOv11-Seg fine-tuned on 506 hand-labelled onboard frames | Deep Learning |
| Data pipeline | pandas ETL + Apache Spark, HDFS, MongoDB | Supporting infrastructure |

Full technical write-up: **[docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)**.
Current limitations are tracked in §8 of that document.
Semester plan and scope decisions: **[docs/ROADMAP.md](docs/ROADMAP.md)**. Planned: Z-score vs
Isolation Forest vs LSTM autoencoder for mistake detection, and Ridge vs RNN vs LSTM for tyre/pace
prediction. Current progress, next steps and the rules for contributing (documentation, testing,
verification) are in §0 and §3 of the roadmap.

---

## 📁 Project Structure

```text
ApexHunter/
├── .github/
│   ├── workflows/ci-cd.yml          # build, checks, unit tests, packaging, optional deployment hook
│   └── dependabot.yml              # weekly GitHub Actions and Docker base-image update PRs
├── .streamlit/config.toml           # Streamlit server configuration
├── .dockerignore                    # excludes secrets, datasets, models and caches from builds
├── Dockerfile                       # Python 3.12 CPU image, non-root user, HTTP health check
├── backend/
│   ├── config.json                  # seasons, CV thresholds, circuits, video URLs
│   └── scripts/
│       ├── utils.py                 # shared paths, config, logger, IST timezone
│       ├── detect_mistakes.py       # Isolation Forest orchestrator
│       ├── train_lstm.py            # LSTM training orchestrator
│       ├── predict_cliff.py         # LSTM prediction orchestrator
│       ├── optimal_line.py          # A*/Dijkstra/BFS orchestrator
│       ├── etl/                     # download_season_data, clean_telemetry, fix_parquet_timestamps,
│       │                            #   download_manual_videos, download_satellite_images
│       ├── bigdata/                 # hdfs_manager, spark_clean_telemetry, mongo_manager
│       ├── cv/                      # extract_frames, select_training_frames, extract_turns,
│       │                            #   run_inference + inference_{geometry,masking,hud,io}
│       ├── isolation_forest/        # mistakes_features, mistakes_model, mistakes_io
│       ├── lstm/                    # tyre_data, tyre_model, tyre_io
│       ├── racing_line/             # racing_line_grid, racing_line_search, racing_line_io
│       └── tests/                   # unit/ (23 files, 127 tests), integration/ (5 files, 18 tests), run_tests.py
├── frontend/
│   ├── app.py                       # Streamlit entry point
│   ├── config.py                    # paths, driver/team mappings, colours
│   └── components/                  # sidebar, data_loader, header_bar, telemetry_charts, track_map,
│                                    #   cv_feed, ai_analysis, racing_line, bigdata_tab, bigdata_charts, scrubber
├── models/                          # best.pt (YOLO), tyre_lstm.pt, tyre_scaler.pkl, tyre_config.json
├── data_lake/                       # data + model outputs (gitignored)
│   ├── season_data/  clean_data/    # raw / cleaned telemetry parquet (92 sessions each)
│   ├── downloaded_videos/ edited_videos/ cv_frames/ cv_dataset/
│   ├── raw_images/satellite_maps/
│   ├── mistake_data/ tyre_predictions/ racing_lines/
│   ├── processed_video/ processed_csv/
│   └── hdfs_status.json  spark_run_log.json  mongo_status.json
├── cache/                           # FastF1 HTTP cache (gitignored)
├── docs/                            # project documentation (+ previous-semester submission files)
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Install

CI and the Docker image use **Python 3.12**. To run the project directly on your computer:

```bash
git clone https://github.com/parikhayush21/ApexHunter.git
cd ApexHunter
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Data pipeline
All scripts can be run by file path from the project root (or as modules from `backend/scripts`,
e.g. `python -m etl.clean_telemetry`):
```bash
python backend/scripts/etl/download_season_data.py      # FastF1 → data_lake/season_data/
python backend/scripts/etl/clean_telemetry.py           # → data_lake/clean_data/
```

### 3. Models
```bash
# Isolation Forest mistake detection
python backend/scripts/detect_mistakes.py --session data_lake/clean_data/2023_1_R.parquet --driver 1

# LSTM tyre cliff (train once, then predict)
python backend/scripts/train_lstm.py --force
python backend/scripts/predict_cliff.py --session data_lake/clean_data/2023_1_R.parquet --driver 1

# Racing line search
python backend/scripts/optimal_line.py --session data_lake/clean_data/2023_1_Q.parquet --driver 1

# YOLO apex detection on an onboard video
# --video-offset = seconds into the video at which the timed lap starts (corners come from FastF1)
python backend/scripts/cv/run_inference.py --input "data_lake/edited_videos/2024/01_bahrain_ver_pole - Trim.mp4" --video-offset 4.9
```

### 4. Dashboard
```bash
streamlit run frontend/app.py
```

### 5. Tests

```bash
python backend/scripts/tests/run_tests.py          # unit tests (fast)
python backend/scripts/tests/run_tests.py --all    # unit + integration (slow)
python backend/scripts/tests/run_tests.py --file test_racing_line_pipeline.py   # one file (integration files run, not skipped)
```

CI runs the unit suite. Real-data integration tests also need the local `data_lake/` and FastF1
`cache/`; they are not included in CI. The pipeline's HTTP check verifies Streamlit startup,
while telemetry, videos and model results require the corresponding data files.

### 6. CI/CD

The [GitHub Actions workflow](.github/workflows/ci-cd.yml) runs automatically after a push to any
branch and when a pull request is opened or updated. It can also be run manually from the
repository's [Actions tab](https://github.com/parikhayush21/ApexHunter/actions/workflows/ci-cd.yml).

| Stage | What happens |
|---|---|
| Build | Creates a Python 3.12 Docker image with CPU-only PyTorch. |
| Check | Runs Ruff fatal-error checks and verifies installed dependency compatibility. |
| Test | Runs the existing unit suite and checks Streamlit's HTTP health endpoint. |
| Publish | After successful checks on a push to `main`, pushes the same tested image to GitHub Container Registry. |
| Deploy | Requests a hosting deployment only when the `DEPLOY_WEBHOOK_URL` repository secret is configured. |

Published images use `ghcr.io/parikhayush21/apexhunter:<full-commit-sha>` and
`ghcr.io/parikhayush21/apexhunter:latest`. A failed check stops publication and deployment.
Pull requests, other branches and manual runs perform checks without publishing an image.

GitHub supplies the registry authentication token automatically. **Hosting is not configured**;
the deployment step records a skip until a hosting webhook is added. For the YAML explanation,
webhook setup, image access and persistent storage, see **[docs/CI_CD.md](docs/CI_CD.md)**.

### 7. Run with Docker

With Docker Engine running, build and launch the dashboard from the repository root:

```bash
docker build -t apexhunter:local .
docker run --rm --publish 8501:8501 apexhunter:local
```

Open [http://localhost:8501](http://localhost:8501). The image excludes the data lake, models and
FastF1 cache. Without cleaned telemetry, the dashboard displays a missing-data message.
To use existing results, mount persistent folders at `/app/data_lake`, `/app/models` and
`/app/cache`; the cache must be writable by the container's user (UID 10001). The
[deployment guide](docs/CI_CD.md#docker-and-persistent-data) includes a complete mount example.

### Optional: Big Data stack
Requires a local Hadoop/HDFS (`hdfs://localhost:9000`) and MongoDB (`localhost:27017`).
```bash
python backend/scripts/bigdata/hdfs_manager.py --upload all --stats
python backend/scripts/bigdata/spark_clean_telemetry.py --hdfs --force
python backend/scripts/bigdata/mongo_manager.py --upload --stats
```

---

## 🎯 Dashboard Features

| Tab | What it shows |
|---|---|
| **Race Intelligence** | Master time scrubber, YOLO HUD video with live apex status and turn report card, speed/mistake track map |
| **Telemetry** | Speed, throttle, brake, gear and RPM traces; top speed and brake/throttle overlap % (samples ≥ 50 km/h) |
| **AI Analysis** | Isolation Forest mistakes per lap; LSTM tyre-cliff chart with confidence band |
| **Racing Line** | Driver's fastest lap vs A\* / Dijkstra / BFS path, deviation per turn |
| **Big Data** | Pipeline DAG, HDFS/Spark statistics, EDA charts, MongoDB leaderboard |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Data | FastF1, yt-dlp, staticmap (Esri imagery), Roboflow (annotation) |
| Processing | pandas, NumPy, PyArrow/Parquet, Apache Spark (PySpark 3.5), HDFS, MongoDB (PyMongo) |
| Machine learning | scikit-learn (Isolation Forest, KFold, StandardScaler), SciPy |
| Deep learning | PyTorch (LSTM), Ultralytics YOLOv11-Seg, OpenCV |
| Frontend | Streamlit, Plotly |
| Testing | unittest, parallel runner |
| CI/CD | GitHub Actions, Docker, GitHub Container Registry, Dependabot |

---

## 👥 Team

| Name | Enrolment No. |
|---|---|
| Parin Shah | 23001091 |
| Ayush Parikh | 23000936 |
| Malav Joshi | 23001119 |
