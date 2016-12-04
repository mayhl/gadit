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
 
## Minimal Setup

To implement GADIT, it is instructive to read the main.cu file for a minimal example of how to initialize and execute GADIT. To implement your own thin film model, you may follow the instructions in the model_default.h file and edit appropriately.  Note that the main.cu file should also be modified to match the new parameters.

GADIT also backs up the solution at a user specified time interval; therefore,  simulations may be stopped and started without any significant loss.

### Input 
The only possible external input to GADIT is initial condition data. The data must be in binary 64-bit float format and place in input/InitialCondition.bin
of the executing directory, or the directory specified by the user. In addition (for now), the data must be padded by two ghost points on  each boundary e.g.  a n by m matrix becomes n+4 by m+4 matrix. There is no need to specify the ghost points.

### Output

Similar to the initial condition data,  the solution at the output times are in binary 64-bit float format and are padded with the ghost points. Solution data will be saved in the sub-folder 'output' of the executing directory or the directory specified by the user. 

In addition to the solution data, several files are created in the 'status' sub-directory.

#### Parameters.txt
A record of the parameters used to initialize GADIT.

#### Status.txt
Useful notifications about the status of GADIT. Primarily contains information of when the simulation was started and last backed up. 
The parameters data structure may be edit so that time step information is also outputted to the file. In addition, the data may also be outputted to the console.

#### Timesteps.bin
Binary 64-bit float data containing the timestep value at every point. 

#### NewtonIterationCounts.bin
Binary  64-bit integer data containing the amount of Newton iterations at each time step. 

#### GADITstatus.bin
Binary 32-bit integer data which records the failure states of Newton iteration scheme.  Also, records time step increases. See namespace newton_status in parameters.h for failure code designations.

#### GADITstatusIndices
Binary 64-bit integer data containing the timestep indices for records in GADITstatus.bin.

#### Temporary Files 
The sub-folder 'temp' contains a backup of solutions. There are two files, BackupSolutionData.bin, the solution data; and BackupSolutionData.bin, a copy of the timestep_manager object.



## Advanced Features

Below is a list of GADIT advanced features.

### Saving Models for Later Use
Instead of altering the default model, you may create a custom entry for your model that may be selected in GADIT by the model ID parameter. 

1) Add entry to enum 'id' in namespace model located in solver_template.h file.
2) Copy model_default and name appropriately. 
3) Rename  MODEL_DEFAULT compiler directive.
4) Rename namespace model_default appropriately.
5) Update 'const model::id ID' to match value created in 1).
6) Change parameters and non-linear function definitions as needed.
7)  Add new switch statement to model_list.h file.
