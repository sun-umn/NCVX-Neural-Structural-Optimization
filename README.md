# Installation

conda create -n ncvx_exp_pami python=3.9

## Install pygranso

    git clone https://github.com/sun-umn/PyGRANSO.git

    cd PyGRANSO

    pip install git+https://github.com/sun-umn/PyGRANSO@dimension_factor

    pip install -r requirements.txt -f https://download.pytorch.org/whl/cu111/torch_stable.html

## (Optional, not used in this src) Install neural-structral-optimization

pip install -q tf-nightly git+https://github.com/google-research/neural-structural-optimization.git

# Problem Description

![Structural_OPT](./structural_opt.png)

The PyGRANSO implementation is based on the MBB beam example of *neural-structral-optimization*. See section **MBB Beam (Figure 2 from paper)** of https://github.com/google-research/neural-structural-optimization/blob/master/notebooks/optimization-examples.ipynb for more details.



# Updates 🎉

1. Stiffness matrix **K** should be a function of grid density *x*. Current implementation uses a fixed constant matrix **K**. Please check the function *objective* in the file *topo_physics.py* in the *neural-structral-optimization* package for more details ✅

	In our original implementation we were creating **K** from `torch` 	as a random initialization. This was done as `K = torch.randn((60 * 	20, 60 * 20))`. **K** is dependent on the *stiffness* which is 	derived from the *young modulus* applied to the phsyical density. 	**K** also depends on the stiffness matrix that we refer to 	currently as **ke** in our code.

2. Current implementation of PyTorch CNN is not a faithful translation of the original TensorFlow version CNN in MBB Beam examle. For example, weight intilizer for the convolution kernel, AddOffSet, latent initializers are not implemented. Please check the file *models.py* in the *neural-structral-optimization* package for more details ✅

	The next item on our **TODO** list was to make sure we had an exact 	implemenation of the `CNNModel`. We went through each layer from 	`tenssorflow` implementation and compared it with each of the 	`torch` layers until we had parity. 

3. Einstein Summation used in the equality constraints *KU = F*. Current implementation of physical law is not the same ✅

	Finally, our implemenation of the compliance and the einstein 	summation were a question mark (❓). Fortunately, `torch` also has 	a version of the `einsum` in their library so we could just 	naturally extend the `numpy` version to `torch`. For added measure 	we also tested the *compliance* calculation from the *Tensorflow* 	implementation and matched against our `torch` calculations.
	

4. We also replicated the `autograd` extensions from the *Tensorflow* implementation. Current, experimentations with `PyGranso` have not yielded any good results so far. The implementation of the `displacement` also now matches exactly so a final question was if an incorrect implementation of bounding the *x* values was an issue. **Still experimenting with this** 🔬.
	
	
# TODO 📝

1. Push up the experimental results with our original implementation.

2. Show the `PyGranso` with all of the brand new updates. Need to clean up the notebook. ✅

3. Show also the `L-BFGS-B` implementation. To see if we can further debug any possible issues 🐞.

4. Add the `tests` directory to show that all our calculations match expectations.

5. I still like the idea of actually solving the constrained problem head on. The newest results are actually based on the full implementations with custom gradients and solvers. I want to be able to show generality for the problem.

6. Add an implementation with `torch` based optimizers. ✅

Test Update Git
