import numpy as np
letters = ['a','b','c','d','e','f','g']

F=np.zeros((200,200,6))

resX=np.zeros((9000,200,200,3))
resY=np.zeros((9000,200,200,6))
for i in range(200):
        for j in range(200):
                F[i][j][0]=F[i][j][1]=F[i][j][2]=1
cnt=0
for i in range(7):
        X = np.load('C:/Users/GuestML/PlastroSentries/Feed/feed ' + letters[i] + '.npy')
        Y = np.load('C:/Users/GuestML/PlastroSentries/Labels/labels ' + letters[i] + '.npy')
        for nn in range(Y.shape[0]):
            tot=np.sum(np.multiply(Y[nn][:][:][:],F[:][:][:]))
            if(tot>180):
                resX[cnt]=X[nn][:][:][:]
                resY[cnt]=Y[nn][:][:][:]
                cnt+=1
        if(cnt>=8999): break
        
resX=resX[:cnt+1][:][:][:]
resY=resY[:cnt+1][:][:][:]
print("non land images" + str(resX.shape[0]))
print("non land labels" + str(resY.shape[0]))
np.save("Xnonland.npy",resX)
np.save("Ynonland.npy",resY)
