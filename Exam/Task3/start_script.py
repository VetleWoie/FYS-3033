import subprocess
import time

model_list = [(0,{"batch_norm":False, "dropout":False,"l2":False},"VGG 11"),
                 (0,{"batch_norm":True, "dropout":False,"l2":False},"VGG 11 with batchnorm"),
                 (0,{"batch_norm":False, "dropout":True,"l2":False},"VGG 11 with dropout"),
                 (0,{"batch_norm":True, "dropout":False,"l2":True},"VGG 11 with batchnorm and l2"),
                 (0,{"batch_norm":True, "dropout":True,"l2":True},"VGG 11 with batchnorm, dropout and l2"),
                 (1,{"batch_norm":False, "dropout":False,"l2":False},"Time distributed VGG 11"),
                 (1,{"batch_norm":True, "dropout":False,"l2":False},"Time distributed VGG 11 with batchnorm"),
                 (1,{"batch_norm":False, "dropout":True,"l2":False},"Time distributed VGG 11 with dropout"),
                 (1,{"batch_norm":True, "dropout":False,"l2":True},"Time distributed VGG 11 with batchnorm and l2"),
                 (1,{"batch_norm":True, "dropout":True,"l2":True},"Time distributed VGG 11 with batchnorm, dropout and l2"),
                ]

for i in range(len(model_list)//2):
    p = subprocess.run(f"python3 task3.py {i}", shell=True)
    # while p.poll():
    #     time.sleep(1)
    time.sleep(10)