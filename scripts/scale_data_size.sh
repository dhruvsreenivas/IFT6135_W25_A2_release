for r_train in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo "============== Running data scaling experiments for LSTM with data fraction $r_train =============="
    python run_exp.py --p 31 --operator + --r_train $r_train --model lstm --optimizer adamw --log_dir /network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/lstm/scale_data/r-train-$r_train --exp_name scale_data_$r_train --multiple
done

for r_train in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    echo "============== Running data scaling experiments for GPT with data fraction $r_train =============="
    python run_exp.py --p 31 --operator + --r_train $r_train --model gpt --optimizer adamw --log_dir /network/scratch/d/dhruv.sreenivas/ift-6135/hw2/logs/gpt/scale_data/r-train-$r_train --exp_name scale_data_$r_train --multiple
done