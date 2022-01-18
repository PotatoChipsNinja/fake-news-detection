for model in bert textcnn bigru eann mdfend my
do
    for i in {1..10}
    do
        python main.py --cuda --gpu 0 --model $model --model-save-dir ./params/params$i  # train
        python main.py --cuda --gpu 0 --model $model --test ./params/params$i/params_$model.pt >> results/$model.txt  # test
    done
done
