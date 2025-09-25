python main.py --dataset cora --epoch 250 --lr 0.001 --alpha 0.1 --beta 1 --gamma 0.5
python main.py --dataset citeseer --epoch 150 --lr 0.001 --alpha 0.4 --beta 2 --gamma 4
python main.py --dataset ACM --epoch 450 --lr 0.0005 --alpha 0.1 --beta 4 --gamma 4
python main.py --dataset citation --epoch 350 --lr 0.001 --alpha 0.1 --beta 0.5 --gamma 0.5
python main.py --dataset DBLP --epoch 350 --lr 0.0005 --alpha 0.1 --beta 1 --gamma 1
python main.py --dataset ogbn-arxiv --epoch 20 --lr 0.001 --alpha 0.1 --beta 4 --gamma 0.5
python main.py --dataset Reddit --epoch 350 --lr 0.0005 --alpha 0.1 --beta 0.5 --gamma 0 --batch_size_sampling 1000
python main.py --dataset YelpChi --epoch 350 --lr 0.001 --alpha 1 --beta 0.1 --gamma 0