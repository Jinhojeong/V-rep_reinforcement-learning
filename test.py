from multiprocessing import Pool
from main import train
if __name__ == '__main__':
    p = Pool(4)
    print(p.map(train, range(19998,20001)))