from multiprocessing import Process
from main import train
if __name__ == '__main__':
    ports = [19998, 19999, 20000]

    procs = []
    for idx, port in enumerate(ports):
        proc = Process(target=train, args=(port,))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()