## Data Generation
In this folder, you will find scripts which were used to generate data from the LISA and T-LISA models. A breif description of each script follows:

- **LISA_Redefined_SIR_Model.py**:
  This script utilizes the mean-field T-LISA rules prior to re-scaling (Honors Thesis EQs 1.8 - 1.11.) The script generates several solutions to the ordinary differential equation formualtion of the T-LISA model through $\gamma$ and $\omega$ space (using scipy.integrate) and stores the resulting solutions list in a collection of CSV files.

- **T-LISA_Paramater_Sweep.py**:
  This script utilizes both the agent-based rules for the T-LISA model (Honors Thesis EQs 2.1 - 2.3) as well as rescaled mean-field rules (Honors Thesis EQs 2.7-2.10.) The scripts sweeps over paramaters $N$, $r$, $\gamma$, $\omega$, and $k$, generating solutions data for all cases. For stochastic (agent-based) simulations, the "batch_size" paramater is set to run multiple simulations. The script also compares re-scaled mean-field predictions with agent-based predictions at each step in paramater space using avergae error (Honors Thesis section 3.1.1.) 
