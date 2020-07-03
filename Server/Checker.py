import numpy as np
letters = ['a','b','c','d','e','f','g']

F=np.load("Ynonland.npy")
num=F.shape[0]
X=F.shape[1]
Y=F.shape[2]
error=0
total=0
for i in range(num):
        for j in range(X):
                for k in range(Y):
                        #print(F[i][j][k][:])
                        if(np.argmax(F[i][j][k][:])>2): error+=1
        if(error>20):
                total+=1
	error=0
	print(total)


print("error total"+str(error))
