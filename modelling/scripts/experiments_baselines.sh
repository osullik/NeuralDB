##
## Copyright (c) 2021 Facebook, Inc. and its affiliates.
##
## This file is part of NeuralDB.
## See https://github.com/facebookresearch/NeuralDB for further info.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##   tsp python src/neuraldb/run.py \
function do_predictions() {
  model_path=$1
  generator=$2
  predictions_path=$3
  python src/neuraldb/run.py \
    --model_name_or_path $model_path \
    --output_dir $model_path \
    --predictions_file $model_path/predictions.jsonl \
    --do_predict --test_file resources/${predictions_path}/test.jsonl \
    --instance_generator $generator \
    --per_device_eval_batch_size 4 \
    --predict_with_generate

}

dataset=${1:-v2.4_25}
export seed=${SEED:-1}

##Removed task spooler (TSP) 

###Group 1 ##

##Original (Commented out) by kent
##SEED=${seed} tsp bash scripts/baselines/train_t5.sh $dataset perfectir 1e-4
##SEED=${seed} tsp bash scripts/baselines/train_t5.sh $dataset wholedb 1e-4

##Changed (COMMENTED OUT SO THEY DON'T RUN AGAIN):
#echo "Training t5 prefect ir"
#SEED=${seed}  bash scripts/baselines/train_t5.sh $dataset perfectir 1e-4
#echo "Training t5 wholedb"
#SEED=${seed}  bash scripts/baselines/train_t5.sh $dataset wholedb 1e-4

#echo "Predicting t5 prefect ir"
do_predictions work/${dataset}/model=t5,generator=perfectir,lr=1e-4,steps=1/seed-${seed} perfectir ${dataset}
echo "Predicting t5 wholedb"
do_predictions work/${dataset}/model=t5,generator=wholedb,lr=1e-4,steps=1/seed-${seed} wholedb ${dataset}

### Group 2 ##

##Original (Commented out) by kent
##SEED=${seed} tsp bash scripts/baselines/train_longformer.sh $dataset perfectir 1e-4
##SEED=${seed} tsp bash scripts/baselines/train_longformer.sh $dataset wholedb 1e-4

##Changed:
#echo "Training Longformer with perfectir"
#SEED=${seed} bash scripts/baselines/train_longformer.sh $dataset perfectir 1e-4
#echo "Training Longformer with wholeDB"
#SEED=${seed} bash scripts/baselines/train_longformer.sh $dataset wholedb 1e-4

echo "Predicting Longformer with perfectir"
do_predictions work/${dataset}/model=longformer,generator=perfectir,lr=1e-4,steps=1/seed-${seed} perfectir ${dataset}
echo "Predicting Longformer with wholeDB"
do_predictions work/${dataset}/model=longformer,generator=wholedb,lr=1e-4,steps=1/seed-${seed} wholedb ${dataset}

## Group 3 ##

##Original (Commented out) by Kent
##SEED=${seed} tsp bash scripts/baselines/train_t5_retriever.sh $dataset externalir dpr 1e-4
##SEED=${seed} tsp bash scripts/baselines/train_t5_retriever.sh $dataset externalir tfidf 1e-4

##Changed:
#echo "Training t5_rx with dpr"
#SEED=${seed} bash scripts/baselines/train_t5_retriever.sh $dataset externalir dpr 1e-4
#echo "Training t5_rx with tfidf"
#SEED=${seed} bash scripts/baselines/train_t5_retriever.sh $dataset externalir tfidf 1e-4

echo "predcting t5_rx with dpr"
do_predictions work/${dataset}/model=t5,generator=externalir,retriever=dpr,lr=1e-4,steps=1/seed-${seed} externalir ${dataset}_dpr
echo "Predicting t5_rx with tfidf"
do_predictions work/${dataset}/model=t5,generator=externalir,retriever=tfidf,lr=1e-4,steps=1/seed-${seed} externalir ${dataset}_tfidf

