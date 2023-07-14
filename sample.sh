python main.py --dataset w7a --method sgd --worker-size 20 --epochs 180 --batch-size 20 --posNum 4 --lr 0.01 --iteration 10000
python main.py --dataset w7a --method coda --worker-size 20 --epochs 180 --batch-size 20 --posNum 4 --lr 0.01 --iteration 10000
python main.py --dataset w7a --method slate --worker-size 20 --epochs 180 --batch-size 20 --posNum 2 --lr 0.01 --thrd 0.1 --iteration 10000
python main.py --dataset w7a --method slatem --worker-size 20 --epochs 180 --batch-size 20 --posNum 2 --lr 0.01 --thrd 0.1 --alpha 0.1 --iteration 10000

python main.py --dataset w8a --method sgd --worker-size 20 --epochs 70 --batch-size 20 --posNum 4 --lr 0.01 --iteration 10000
python main.py --dataset w8a --method coda --worker-size 20 --epochs 70 --batch-size 20 --posNum 4 --lr 0.01 --iteration 10000
python main.py --dataset w8a --method slate --worker-size 20 --epochs 80 --batch-size 20 --posNum 2 --lr 0.01 --thrd 0.1 --iteration 10000
python main.py --dataset w8a --method slatem --worker-size 20 --epochs 80 --batch-size 20 --posNum 2 --lr 0.01 --thrd 0.1 --alpha 0.1 --iteration 10000

python main.py --dataset mnist --method sgd --worker-size 20 --epochs 5 --batch-size 20 --posNum 4 --lr 0.01 --iteration 300
python main.py --dataset mnist --method coda --worker-size 20 --epochs 5 --batch-size 20 --posNum 4 --lr 0.01 --lr2 0.001 --iteration 300
python main.py --dataset mnist --method slate --worker-size 20 --epochs 5 --batch-size 20 --posNum 3 --lr 0.01 --thrd 0.1 --iteration 300
python main.py --dataset mnist --method slatem --worker-size 20 --epochs 5 --batch-size 20 --posNum 3 --lr 0.01 --thrd 0.1 --alpha 0.9 --iteration 300

python main.py --dataset fmnist --method sgd --worker-size 20 --epochs 5 --batch-size 20 --posNum 4 --lr 0.005 --iteration 300
python main.py --dataset fmnist --method coda --worker-size 20 --epochs 10 --batch-size 20 --posNum 4 --lr 0.005 --lr2 0.0005 --iteration 300
python main.py --dataset fmnist --method slate --worker-size 20 --epochs 4 --batch-size 20 --posNum 5 --thrd 0.1 --lr 0.005 --iteration 300
python main.py --dataset fmnist --method slatem --worker-size 20 --epochs 5 --batch-size 20 --posNum 5 --thrd 0.1 --lr 0.005 --alpha 0.9 --iteration 300

python main.py --dataset cifar10 --method sgd --worker-size 20 --epochs 1000 --batch-size 60 --lr 0.01 --iteration 2000
python main.py --dataset cifar10 --method coda --worker-size 20 --epochs 200 --batch-size 60 --lr 0.01 --lr2 0.001 --iteration 2000 
python main.py --dataset cifar10 --method slate --worker-size 20 --epochs 200 --batch-size 60 --posNum 20 --lr 0.01 --thrd 0.1 --iteration 2000 
python main.py --dataset cifar10 --method slatem --worker-size 20 --epochs 1000 --batch-size 60 --posNum 25 --lr 0.01 --thrd 0.1 --alpha 0.1 --iteration 2000 > 0.txt

