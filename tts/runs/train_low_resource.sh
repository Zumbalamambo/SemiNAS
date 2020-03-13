cd ..
MODEL=low_resource
OUTPUT_DIR=outputs/$MODEL
DATA_DIR=data/LJSpeech-1.1/processed_low_resource_1k5

export PYTHONPATH=.:$PYTHONPATH

sudo apt-get update
sudo apt-get install libsndfile1 -y

mkdir -p $OUTPUT_DIR

python train.py \
  --data=$DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --arch="10 10 4 3 6 3 9 2 4 2 11 6" \
  | tee -a $OUTPUT_DIR/train.log
