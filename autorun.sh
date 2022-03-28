model=(bert bertExt textcnn textcnnExt bigru bigruExt eann eannExt mdfend mdfendExt)
lr=(0.0007 0.0007 0.0003 0.0003 0.0003 0.0003 0.0007 0.0007 0.0005 0.0005)
seed=(412 643 234 43 12 65 23 98 50 9)
for repeat in {1..10}
do
    rm ./data/dataloader.pkl
    for i in {0..9}
    do
        python main.py --cuda --gpu 0 --model ${model[i]} --lr ${lr[i]} --batch-size 64 --seed ${seed[repeat]} --model-save-dir ./params/params$repeat  # train
        python main.py --cuda --gpu 0 --model ${model[i]} --test ./params/params$repeat/params_${model[i]}.pt >> results/${model[i]}.txt  # test
    done
done
