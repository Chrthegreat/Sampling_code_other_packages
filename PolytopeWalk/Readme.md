# Package PolytopeWalk

🔗 **Repository:** https://github.com/ethz-randomwalk/polytopewalk

*In this folder you can find the code used for the barrier walks using the PolytopeWalk Python package.*

---

## Files Provided

### Diagnostics
Holds the functions for finding the ESS and PSRF metrics.  
The initial thought was to use `arviz` like the authors did, but I ended up copying the Volesti versions in order to match their results. Since some variations were noted between implementations, I ultimately used the Volesti copies for the final results.

### Generators
This file holds functions that generate polytopes for use in the sampling code.  
There are many variations, even of the same polytope, but the most used ones are those copied from the Volesti generators.  

Unfortunately, for the sparse code, the Birkhoff polytope had to be defined in equation form (like in the provided tests), since the other definitions wouldn’t work.

### Main Dense
Holds the logic for the dense walks.  
You can choose the desired polytope, batch size, ESS target, thinning factor, walk radius, upper time limit, and other parameters.

### Main Sparse
Holds the logic for the sparse walks.  
Same as the dense version — it offers many customization options.

---

**Note:**  
Sparse methods for the Birkhoff polytope managed to sample up to 144 dimensions, while the dense ones took too long even for 25 dimensions.
