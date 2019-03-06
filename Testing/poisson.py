import math

def probs(scale):
     arr = []
     cumPr = math.exp(-scale) * scale ** (0)
     factor = 1
     factorCounter = 1
     newFires = 0
     arr.append(cumPr)

     for ii in range(1,50):
             newFires += 1
             cumPr += math.exp(-scale) * scale ** (newFires) / factor
             factorCounter += 1
             factor = factor * factorCounter
             arr.append(cumPr)
     return(arr)


arr = probs(1.5)
print(arr)
