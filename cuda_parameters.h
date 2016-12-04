// ----------------------------------------------------------------------------------
// Copyright 2016-2017 Michael-Angelo Yick-Hang Lam
//
// The development of this software was supported by the National Science Foundation
// (NSF) Grant Number DMS-1211713.
//
// This file is part of GADIT.
//
// GADIT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as published by
// the Free Software Foundation.
//
// GADIT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GADIT.  If not, see <http://www.gnu.org/licenses/>.
// ----------------------------------------------------------------------------------

// ----------------------------------------------------------------------------------
// Name:			cuda_parameters.h
// Version: 		1.0
// Purpose:		Contains parameters defining spatial sub-domain sizes
//						and dimensions of grid of sub-domains for CUDA 
//						parallelization. Parameters are tuned to match a 
//						specific GPU.
// ----------------------------------------------------------------------------------

#ifndef CUDA_PARAMETERS
#define CUDA_PARAMETERS

#include "work_space.h"
#include "dimensions.h"

namespace cuda_parameters
{ 
	
	// pads cells with additional points which will be set to satisfy boundary conditions.
	// DO NOT CHANGE
	int const CELL_BORDER_PADDING = 2;

	int const DEBUG_MODE = false;

	// Parameters that you may change with out introducing an error.
	// Only multiples of 8 should be used.
	// Can be used to tune CUDA code to a specific GPUs
	int const PENTA_LU_LINE_THREAD_SIZE = 16;
	
	int const SIMPLE_SQUARE_BLOCK = 16;

	int const SOLVE_JY_THREAD_SIZE = 32;

	int const REDUCTION_BLOCK_SIZE = 256;


	// Should be no greater than 15
	int const SOLVE_JY_SUBLOOP_SIZE = 15;


	struct kernal_launch_parameters_pair{
		dim3 thread;
		dim3 block;
	};

	// Contains information about sub domain 
	class kernal_launch_parameters
	{

	public:

		kernal_launch_parameters_pair penta_x_direction;
		kernal_launch_parameters_pair penta_y_direction;


		kernal_launch_parameters_pair simple_sqaure_block;
		kernal_launch_parameters_pair simple_sqaure_block_transpose;

		kernal_launch_parameters_pair padded_sqaure_block_transpose;

		kernal_launch_parameters_pair compute_Jx;
		kernal_launch_parameters_pair compute_Jy;

		kernal_launch_parameters(){};

		void initalize( dimensions dims)
		{

			// For linear solver, subdivide domain into strips in
			// direction of ADI step and assign a 1D array of threads 
			// to stripe
			int thread_line_size = PENTA_LU_LINE_THREAD_SIZE;
			int thread_block_size = PENTA_LU_LINE_THREAD_SIZE;

			int blocksPerGrid_x = (dims.n + thread_block_size - 1) / thread_block_size;
			int blocksPerGrid_y = (dims.m + thread_block_size - 1) / thread_block_size;

			penta_y_direction.block = dim3(blocksPerGrid_x, 1, 1);
			penta_y_direction.thread = dim3(thread_line_size, 1, 1);

			penta_x_direction.block = dim3(blocksPerGrid_y, 1, 1);
			penta_x_direction.thread = dim3(thread_line_size, 1, 1);


			// Divides domain (without ghost points) into square plots
			// and assigns 2D array of threads.
			blocksPerGrid_x = (dims.n + SIMPLE_SQUARE_BLOCK - 1) / SIMPLE_SQUARE_BLOCK;
			blocksPerGrid_y = (dims.m + SIMPLE_SQUARE_BLOCK - 1) / SIMPLE_SQUARE_BLOCK;

			simple_sqaure_block.block = dim3(blocksPerGrid_x, blocksPerGrid_y, 1);
			simple_sqaure_block.thread = dim3(SIMPLE_SQUARE_BLOCK, SIMPLE_SQUARE_BLOCK, 1);
			simple_sqaure_block_transpose.block = dim3(blocksPerGrid_y, blocksPerGrid_x, 1);
			simple_sqaure_block_transpose.thread = dim3(SIMPLE_SQUARE_BLOCK, SIMPLE_SQUARE_BLOCK, 1);

			// Divides padded domain (with ghost points) into square blocks
			// and assigns 2D array of threads.
			blocksPerGrid_x = (dims.n_pad + SIMPLE_SQUARE_BLOCK - 1) / SIMPLE_SQUARE_BLOCK;
			blocksPerGrid_y = (dims.m_pad + SIMPLE_SQUARE_BLOCK - 1) / SIMPLE_SQUARE_BLOCK;

			padded_sqaure_block_transpose.block = dim3(blocksPerGrid_y, blocksPerGrid_x, 1);
			padded_sqaure_block_transpose.thread = dim3(SIMPLE_SQUARE_BLOCK, SIMPLE_SQUARE_BLOCK, 1);

			// Divides domain into rectangles and assigns 1D array of threads
			blocksPerGrid_x = (dims.n + SOLVE_JY_THREAD_SIZE - 1) / SOLVE_JY_THREAD_SIZE;
			blocksPerGrid_y = (dims.m + SOLVE_JY_SUBLOOP_SIZE - 1) / SOLVE_JY_SUBLOOP_SIZE;

			compute_Jy.block = dim3(blocksPerGrid_x, blocksPerGrid_y, 1);
			compute_Jy.thread = dim3(SOLVE_JY_THREAD_SIZE, 1, 1);

		}

	};
}


#endif