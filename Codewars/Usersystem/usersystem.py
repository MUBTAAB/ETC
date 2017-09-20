# TODO: create the User class
# it must support rank, progress, and the inc_progress(rank) method
rank_dict = [-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8]

class User:
    def __init__(self, rank = -8, progress = 0):

        self.rank = rank
        self.inside_rank = rank_dict.index(self.rank)
        self.progress = progress 
    
    def check_progress(self):
        if self.progress >= 100 and self.rank < 8:
            self.inside_rank += ((self.progress-self.progress%100)/100)
            self.rank = rank_dict[int(self.inside_rank)]
            if self.rank == 8:
                self.progress = 0
            else:
                self.progress = self.progress%100
            print('progressed to rank ' + str(self.rank))
            print('new process: '+ str(self.progress))
            
    def inc_progress(self, rank):
        print('progressing task with ' + str(rank) + ' rank from own rank of: ' + str(self.rank)) 
        print('initial progress: '+ str(self.progress))
        rank = rank_dict.index(rank)
        
        if rank < self.inside_rank-2 or self.rank == 8:
            return
            
        if rank < self.inside_rank:
            self.progress += 1
            self.check_progress()
            return
        
        if rank == self.inside_rank:
            self.progress += 3
            self.check_progress()
            return
        
        if rank > self.inside_rank:
            d = rank-self.inside_rank
            self.progress += 10 * d * d
            self.check_progress()
            return