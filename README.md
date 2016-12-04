# GADIT
GPU Alternating Direction Implicit Thin film solver

## About
GADIT is a GPU implementation of the numerical scheme for the generalize thin film model presented by Witelski and Bowen (2003) (doi).  Comparisons to a similarly coded serial CPU method show that GADIT performs 10-30 times faster, allowing domains containing several millions of points to be solved within a week. The performance results of GADIT are currently in preparation for publication.

## Operating System Support
GADIT was developed and tested on the Windows platform.

Durning early development, GADIT has successfully complied on Ubuntu 14.05; however, there were file I/0 errors. Brief testing showed that by removing the file I/0,  GADIT  ran without issue and produced the correct results (console output).  Since then, the file I/0 has been redesigned; however, due to a technical failure, my Ubuntu 14.05 boot failed, and I have been unable to get NVIDIA driver to work again; therefore, I have been unable to test the new file I/0. The code should work on Linux base operating systems, and if there are any runtime errors, this is most likely due to the file I/0; This can be corrected by modifying output.h. In addition, the original makefile was lost.

GADIT has not been tested on Macintosh based operating systems.

## Known Issues

There is a small bug in the adaptive time stepping method, timestep_manager.h.  GADIT outputs the solution at user specified time steps, 'dt_out,' and will adjust the current time step, 'dt,'  to
calculate the solution at the output time, 't_out,' if 'dt' advances the solution past 't_out.'  GADIT will also increase 'dt' if a certain amount of
time steps are computed within the convergence conditions of the Newton iterations method, 'stable_time_steps.'  There is, however, a bug where both conditions are met, which results in the time step being increased instead; this is an unexpected state; therefore GADIT will terminate.  This issue appears only to occur if 'stable_time_steps' is small, and setting 'stable_time_steps' to a large value e.g. 500 removes this issue. Large values for 'stable_time_steps' should be preferred in any case, as for numerical efficiency as with small 'stable_time_steps,' it is more likely that time is wasted on computations that are not used. 
 
## Getting Started
