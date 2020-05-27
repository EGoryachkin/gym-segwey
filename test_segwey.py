import segwey
import matplotlib.pyplot as plt

env = segwey.segwey()

s = env.reset()
S = []
A = []
d = False
while True:
	a = env.tet.dot(s)
	s, r, d, _ = env.step(a)
	S.append(s)
	A.apend(a)
	if d:
		break
plt.plot(range(len(S)), S)