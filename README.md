KTH 3

Machine Learning for Fast Beam Alignment in Wireless Communications

Chen Chen, Member, IEEE

I. INTRODUCTION

*z* (m)![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.001.png)

*y* (m)![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.002.png)

BS ![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.003.png) User

*x* (m)

Fig. 1. System model.

Notations: Scalars, vectors and matrices are represented by italic letters, boldface lower-case letters, and boldface upper-case letters, respectively. (·)T and (·)H are the transpose and conjugate transpose operations, respectively. diag(a) is a diagonal matrix whose diagonal elements are the corresponding elements in vector a. CN(µ,σ2) stands for a circularly symmetric complex Gaussian distribution with mean µ and variance σ2. ℜ{a} and ℑ{a} denote the real part and imaginary part of a complex number a, respectively. CM ×N and RM ×N represent the space of complex-valued and real-valued matrices, respectively.

1. System Model

As shown in Fig. 1, we consider a multi-antenna base station (BS) and a single-antenna user. The BS is equipped with M antennas. The BS selects a beam from a codebook W with M candidate beams. The greedy search method requires M beam searches (time slots) to find the best beam that provides the maximum transmission rate. We aim to propose a machine learning-

based algorithm to reduce the beam alignment time.

1) Channel Model: The wireless channel between the BS and the user is denoted by h ∈ √ ![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.004.png)

CM ×1. We have h = βg, where β is the large-scale path loss and g is the small-scale fading vector.

The large-scale path loss in dB is computed by

βdB = β0 − 10αlog10 (d) , (1)

where β0 = −30 dB is the path loss at the reference distance, α = 2.2 is the path-loss exponent, d is the distance between the BS and the user.

The small-scale fading is assumed to be Rician fading and is denoted by

κ 1![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.005.png)![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.006.png)

g = 1 + κgL + 1 + κgNL, (2) where κ is the Rician factor, which is set to 10. gL ∈CM ×1 is the line-of-sight (LOS) component

given by

g = 1,e j2πdB cos(ϕ),···,e j2π(M −1)dB cos(ϕ) T , (3)

L λ λ

where λ is the wavelength, dB is the antenna spacing, which is set to λ/2, and ϕ is the azimuth angle between the BS and the user. gNL ∈CM ×1 is the non-line-of-sight (LOS) component model as Rayleigh fading consisting of uncorrelated CN(0,1) elements.

2) Codebook: The codebook is given by W = {w1,w2,···,wM }, in which the m-th beam

is given by

w = ~~√~~1 1,e j2πd ( 2π(m−1) ),···,e j2π(M −1)dB ( 2π(m−1) ) T . (4)![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.007.png)

B cos cos

m λ M λ M

M

3) Objective: For a given wireless channel h, find the optimal beam from W that maximizes

the transmission rate. When w is selected, the transmission rate is calculated as

Rm(w ) = log2 1 + Pt hδ wm 2 ,

`  `H

m 2 (5) where![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.008.png) Pt is the transmit power at the BS, and δ2 is the power of noise.

2. Machine Learning Solution

The idea is that the user transmit T, T ≪ M, pilot signals to the BS, and the BS uses the received signals as input to the neural network. In this way, the beam search overhead reduces from M to T. The output of the neural network is the selected beam index.

In time slot t, 1 ≤ t ≤ T, the user transmits pilot signal s ∈ C1×1 to the BS. We have

t

sH st = 1. The received signal at the BS is

t

yt = PpilotwtH hst + wtH nt, (6) where![](Aspose.Words.af5c9db4-d55b-4d0f-81bf-3025e3a0f9fb.009.png) Ppilot is the pilot power and n ∈ CM ×1 is the noise vector consisting of uncorrelated

t

CN(0,δ2) elements.

The collected received signals [y1,y2,···,yT ]T is used as input of the neural network.
