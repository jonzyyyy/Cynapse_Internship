pip install mlflow
pip install dvc
pip install ultralytics==8.3.26
cd /app
yolo settings mlflow=false
python /app/train.py
# python /app/move.py