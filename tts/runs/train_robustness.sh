cd ..
MODEL=robustness
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/LJSpeech-1.1/processed_robust100

export PYTHONPATH=.:$PYTHONPATH

sudo apt-get update
sudo apt-get install libsndfile1 -y

mkdir -p $OUTPUT_DIR

python train.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --arch="8 6 3 2 11 7 9 1 4 11 6 2" \
  | tee -a $OUTPUT_DIR/train.log
