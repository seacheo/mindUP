import os
import queue
<<<<<<< HEAD

=======
from threading import Thread
import sys
sys.path.append('/network/lustre/iss01/apps/lang/anaconda/3/5.1.0/lib/python3.6/site-packages')
>>>>>>> 4fe51bb9a861ce8152e5a72a6ef1be44d3525287
l = [i for i in range(1000)]


baseFolder='one/'
files=[f for f in os.listdir(baseFolder) if not f.startswith('.')]
# models=5
<<<<<<< HEAD
gpus=4
q = queue.Queue()
q.queue = queue.deque(files)
=======

gpus=4
q = queue.Queue()
# q.queue = queue.deque(files)
[q.put(i) for i in files]
>>>>>>> 4fe51bb9a861ce8152e5a72a6ef1be44d3525287

print('queue is empty', q.empty())

def startChild(g):
    models=5
<<<<<<< HEAD
    while not q.empty:
=======
    while not q.empty():
>>>>>>> 4fe51bb9a861ce8152e5a72a6ef1be44d3525287
        file = q.get()
        for i in range(models):
            os.system('python doWork.py '+ file + ' ' + str(g) + ' ' + str(i))

workers=[]
for i in range(gpus):
    workers.append(Thread(target = startChild, args=(i,)))
for worker in workers:
    worker.start()