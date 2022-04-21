from itertools import permutations
import numpy as np

class PermutationIterator():
    def __init__(self, permutation) -> None:
        self._permutation = permutation
        self._index = 0
    
    def __next__(self):
        print("Iter",self._index,"len:" ,len(self._permutation))
        if self._index >= len(self._permutation):
            print("stop!")
            raise StopIteration
        else:
            result = self._permutation[self._index]
            self._index += 1
            return result

class Permutation():
    def __init__(self, max) -> None:
        self.max = max
        self.perms = [perm for perm in permutations(np.arange(max))]
        self._index = 0 

    def get_index(self, perm):
        perm = tuple(perm)
        for i, p in enumerate(self.perms):
            if p == perm:
                return i

        raise ValueError("No such permutation")

    def __len__(self):
        return len(self.perms)
    
    def __getitem__(self, key):
        return self.perms[key]

    def __iter__(self):
        return PermutationIterator(self)
    
    def __str__(self) -> str:
        return str(self.perms)