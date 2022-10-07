import numpy as np
import matplotlib.pyplot as plt

lut = []

RES = 16

for i in range(RES):
    lut.append(np.sin(i/(RES-1) * np.pi / 2.))

lut = np.array(lut)

print("float sin_luts[] = float[](")
for i in lut:
    print("\t" + str(round(i,20)) + ",")
print(");")

def SinHalfPi(x):
    x = x / (np.pi / 2.) * (RES-1)
    fx = np.floor(x)
    cx = fx + 1
    t = x - fx
    return (1 - t) * lut[fx.astype('int32')] + t * lut[cx.astype('int32')] #np.interp(x, (fx, lut[fx.astype('int32')]), (cx, lut[cx.astype('int32')])

def Sin(x):
    x = np.fmod(x, 2*np.pi)
    ges = np.sign(np.fmod(x, np.pi) - np.pi/2)
    bes = np.sign(np.pi - x)
    x = np.fmod(x, np.pi/2)
    ges2 = (np.pi/2 * (ges > 0)) + ges * -x
    return bes * SinHalfPi(ges2)


def Cos(x):
    idx = np.fmod(x + (1000 * 2 * np.pi + np.pi/2), np.pi*2)
    return lut[np.array(idx*1000, dtype='int32')]

x = np.arange(0, 2 * np.pi, 0.0001)


true = np.sin(x)
luts = Sin(x)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, true, color='blue', label='Sine wave')
plt.plot(x, luts, color='red', label='Sine wave (approx.)')
#plt.xlim((0, 2 * np.pi))
plt.show()
