# Boundary_Equilibrium_GAN
1.[BEGAN](https://arxiv.org/pdf/1703.10717.pdf)의 contribution
======================
* Simple / Robust structure (기존 GAN과 비슷), fast and stable convergence  
* __Equilibrium Concept__으로 discriminator 와 generator의 balance를 조절 가능하다(보통은 training 초기에 discriminator가 쉽게 이기는 문제가 있다)  
* __image diversity__ & __visual quality__ 간의 trade-off를 조절할 수 있는 새로운 방법을 제시했다.  
* convergence 의 approximate measure (global measure)를 제시했다.

2.Proposed Method
====================

* 기존의 GAN은 sample간의 data distribution을 직접적으로 match하려고 하는 반면, 이 논문에서는 auto-encoder loss distribution 간의 match를 하고자 했다. 이 때 Wasserstein distance가 이용된다.  
* gamma = G(z) 와 x 의 expectation error 간의 비율 , 이것을 이용해 image diversity / quality 간의 trade-off를 조절할수 있다. See more in [BEGAN](https://arxiv.org/pdf/1703.10717.pdf)
