# Package PolytopeSampler

🔗 **Repository:** https://github.com/ConstrainedSampler/PolytopeSamplerMatlab

*In this folder you can find the code used for the CRHMC walk using the PolytopeSampler Matlab package.*

---

## Files Provided

### Biology Native
This file located the Recon models inside their folder and uses them to sample. 
It is just a copy paste for an example already provided by the authors.
You have to change the recon number in the file name to choose Recon1, 2 or 3.

### Netlib Native
Same as biology native, we found the Netlib folder and sampled from those polytopes.

### Birkhoff
Code used to sample from the Birkhoff. The polytope is created exactly like the example. 
We just add some extra options like selecting dimensions and target ess and output messages.

### Extract Mat to csv
The main idea for this code was to extract the polytopes from the MatLab version and then process them at will.
Each mat file holds a 'problem' struct with 2 matrices describing the polytope in equalities form along with 2 boundaries matrices.
The null sapce method was used to convert the polytope to the inequalities form expected by the rest of the samplers. 
The outputs were first simplified using Polyround and then rounded using Dingo. Check their folders for more info.