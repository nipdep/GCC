load_path=$1
ARGS=${@:2}

for dataset in $ARGS
do
    python generate.py --dataset $dataset --load-path $load_path
done
