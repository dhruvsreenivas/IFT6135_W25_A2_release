for wd in 0.25 0.5 0.75 1.0
do
    echo "============== Running regularization for LSTM (weight decay $wd) =============="
    python run_exp.py --p 31 --operator + --weight_decay $wd --n_steps 40001 --model lstm --optimizer adamw --log_dir /network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/lstm/regularization/weight-decay-$wd --exp_name regularization_$wd --multiple
done