import os
import queue

l = [i for i in range(1000)]


baseFolder='one/'
files=[f for f in os.listdir(baseFolder) if not f.startswith('.')]
# models=5
gpus=4
q = queue.Queue()
q.queue = queue.deque(files)

print('queue is empty', q.empty())

def startChild(g):
    models=5
    while not q.empty:
        file = q.get()
        for i in range(models):
            os.system('python doWork.py '+ file + ' ' + str(g) + ' ' + str(i))

workers=[]
for i in range(gpus):
    workers.append(Thread(target = startChild, args=(i,)))
for worker in workers:
    worker.start()