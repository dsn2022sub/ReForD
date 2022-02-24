proj_dir=$(pwd)
dataset='wget'
normal_size=125
attack_size=25

cd data || return
mkdir -p normal && mkdir -p attack
mkdir -p maps && mkdir -p straces
cd "$proj_dir" || return

mkdir -p data/normal/base && mkdir -p data/normal/stream
mkdir -p data/attack/base && mkdir -p data/attack/stream
mkdir -p data/normal/filtered-base && mkdir -p data/normal/filtered-stream
mkdir -p data/attack/filtered-base && mkdir -p data/attack/filtered-stream

# do clean before start
# we do not clean after execution, because we assume that the sketches may be used for further analysis
rm -rf evaluators/unicorn/"$dataset"-sketch

mkdir -p "$dataset"-sketch &&
  mkdir -p "$dataset"-sketch/normal &&
  mkdir -p "$dataset"-sketch/attack
# normal
for ((i = 0; i < normal_size; ++i)); do
  bin/unicorn/main filetype edgelist \
  base "$dataset"/normal/filtered-base/base-"$dataset"-normal-"$i".txt \
  stream "$dataset"/normal/filtered-stream/stream-"$dataset"-normal-"$i".txt \
  sketch "$dataset"-sketch/normal/sketch-"$dataset"-normal-"$i".txt \
  decay 500 lambda 0.02 batch 3000 chunkify 1 chunk_size 50
  rm -rf "$dataset"/data/normal/filtered-base/base-"$dataset"-normal-"$i".txt.*
  rm -rf "$dataset"/data/normal/filtered-base/base-"$dataset"-normal-"$i".txt_*
done
# attack
for ((i = 0; i < attack_size; ++i)); do
  bin/unicorn/main filetype edgelist \
  base "$dataset"/attack/filtered-base/base-"$dataset"-attack-"$i".txt \
  stream "$dataset"/attack/filtered-stream/stream-"$dataset"-attack-"$i".txt \
  sketch "$dataset"-sketch/attack/sketch-"$dataset"-attack-"$i".txt \
  decay 500 lambda 0.02 batch 3000 chunkify 1 chunk_size 50
  rm -rf "$dataset"/data/attack/filtered-base/base-"$dataset"-attack-"$i".txt.*
  rm -rf "$dataset"/data/attack/filtered-base/base-"$dataset"-attack-"$i".txt_*
done
