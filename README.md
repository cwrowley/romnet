Yu Shuai's branch of modified ROMNET codes for numerical simulation of stochastic Barkley's model equations using RK3CN, SETD1 and other schemes.

Some newly-added features are:
1. stochastic exponential time differencing (SETD) scheme as a new time-stepper class, see timestepper.py in romnet/src/romnet
2. initialization and instantiation of the SETD time-stepper in the semi-linear model, see model.py in romnet/src/romnet
3. Barkley's turbulent puff model as a new instance of the semi-linear model, see Barkley.py in romnet/src/romnet/models/Barkley.py
4. The main program for numerical simulation of Barkley's model, see Barkley.py in romnet/src/Barkley.py and its corresponding initial conditions as txt document.

P.S. For stochastic simulation with multiplicative/additive noise, we suggest using SETD scheme which addresses the noise term easily. For other deterministic simulation one is free to choose between traditional explicit/semi-implicit schemes and SETD. 
