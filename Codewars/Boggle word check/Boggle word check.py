import numpy as np
import copy

def find_word(board, word):
    print('START '+word+ ' in board:')
    for i in board:
        print(i)
    # write your code here
    def check_neigh(iBoard,x,y,letter):
        fCond = lambda x,y: all([len(board[0])-1 >= x,x >= 0,len(board)-1 >= y,y >= 0]) 
        return([(x+ix,y+iy) for iy in [0,1,-1] for ix in [0,1,-1] if fCond(x+ix,y+iy) and board[x+ix][y+iy] == letter])
        
    for x in range(len(board)):
        print(x)
        for y in range(len(board[0])):
            print('')
            print('New start',x,y)
            x1 = x
            y1 = y
            nth = 0
            iBoard = copy.deepcopy(board)
            mlist = []
            boards = [copy.deepcopy(iBoard)]
            while True: 
                print([x1,y1,word[nth]])
                if len(check_neigh(iBoard,x1,y1,word[nth])) > 0:
                    
                    mlist.append(check_neigh(iBoard,x1,y1,word[nth]))
                    
    
                    x1 = mlist[-1][0][0]
                    y1 = mlist[-1][0][1]
                    print(['Ok',x1, y1])
                    print('Del '+word[nth]+' from pos '+str([x1,y1]))
                    nth += 1
                    iBoard[x1][y1] = np.NaN
                    boards.append(copy.deepcopy(iBoard))
                    for i in iBoard:
                        print(i)
                    del(mlist[-1][0])
                    for i in mlist:
                        print(i)
                else:
                    
                    print('Not found')
                    for i in mlist:
                        print(i)
                    bPass = False
                    while len(mlist) > 0:
                        if mlist[-1] == []:
                            del(mlist[-1])
                            boards = boards[0:len(mlist)]+1
                            nth = len(mlist)-1
                            print('Go back to letter: '+ word[nth])
                        else:    
                            boards = boards[0:len(mlist)]+1
                            iBoard = boards[-1]
                            x1 = mlist[-1][0][0]
                            y1 = mlist[-1][0][1]
                            nth = len(mlist)-1
                            del(mlist[-1][0])
                            print('Step back!' + 'New letter: ' + word[nth] + ' New board:')
                            for i in iBoard:
                                print(i)
                            print(word[nth], [x1,y1])
                            bPass = True
                            break
                            
                    if bPass == False:
                        break
                        
                if nth == len(word):
                    print('word in text, break')
                    return(True)
    print('Out of iterations')
    return False
len([])
