
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
// Name:		newton_iterative_method.h
// Version: 	1.0
// Purpose:		Applies Newton iteration for a given times step. Returns 
// ----------------------------------------------------------------------------------

#ifndef NEWTON
#define NEWTON

#include <chrono>
#include "dimensions.h"
#include "cuda_parameters.h"

#include "nonlinear_penta_solver.h"
#include "compute_Jy_F.h"
#include "compute_Jx.h"

#include "model_list.h"
// List of Models 
#include "model_default.h"
#include "model_nlc.h"



namespace  newton_iterative_method
{
	template <typename DATATYPE, model::id MODEL_ID> __device__ __forceinline__
		void compute_nonlinear_functions(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h , spatial_parameters<DATATYPE> spatialParas, model_parameters<DATATYPE, MODEL_ID> modelParas, DATATYPE dt, int index )
	{

		//switch (MODEL_ID)
		//{
		//case model_id::DEFAULT:
		//	model_default::nonlinear_functions<DATATYPE>(d_ws, d_h , modelParas, index);
		//	break;
		//case model_id::NLC:
		//	model_nlc::nonlinear_functions<DATATYPE>(d_ws, d_h , modelParas, index);
		//	break;

		//}

		model_list::select_functions<DATATYPE,MODEL_ID>( d_ws , d_h , modelParas , index );

		// Note: Recall to multiple function by dt and scaled 1/dx^4 and 1/dx^2 
		d_ws.f1[index] *= dt*spatialParas.scaled_inv_ds4;
		d_ws.df1[index] *= dt*spatialParas.scaled_inv_ds4;

		d_ws.f2[index] *= dt*spatialParas.scaled_inv_ds2;
		d_ws.df2[index] *= dt*spatialParas.scaled_inv_ds2;
	}

		template <typename DATATYPE, bool FIRST_NEWTON_ITERATION> __global__
		void check_blocks(reduced_device_workspace<DATATYPE> d_ws, dimensions dims , newton_parameters<DATATYPE> newt_paras )
	{

		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		DATATYPE *d_h;
		(FIRST_NEWTON_ITERATION) ? (d_h = d_ws.h) : (d_h = d_ws.h_guess);

		if (x < dims.n && y < dims.m)
		{

			int index = (y + dims.padding)*dims.n_pad + x +  dims.padding;
			int index_non_padded = (y)*dims.n + x;


			bool thread_check;

			// To implement more checks on Newton iterative method: 
			//		1) copy next three lines, and alter appropriately 
			//		2) Add new enum value to newton_status::status in parameters.h
			//		3) add need ifelse statement before newton_status::SUCCESS

			__shared__ bool block_newton_convergence_check;
			thread_check = (abs(d_ws.v[index_non_padded] / d_h[index]) > newt_paras.error_tolerence);
			block_newton_convergence_check = __syncthreads_or(thread_check);


			d_ws.h_guess[index] = d_h[index] + d_ws.v[index_non_padded];


			if (threadIdx.x == 0 && threadIdx.y == 0)
			{

				int reduced_index = blockIdx.y*blockDim.x + blockIdx.x;

				if (block_newton_convergence_check)
					d_ws.solution_flags[reduced_index] = newton_status::CONVERGENCE_FAILURE_LARGE;
				else
					d_ws.solution_flags[reduced_index] = newton_status::SUCCESS;

			}
		}

	}
	template <typename DATATYPE, bool FIRST_NEWTON_ITERATION> __global__ void update_guess(reduced_device_workspace<DATATYPE> d_ws, dimensions dims)
	{

		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;
		int index = (y + dims.padding)*dims.n_pad + x + dims.padding;
		int index_non_padded = (y)*dims.n + x;
		
		DATATYPE *d_h;
		(FIRST_NEWTON_ITERATION) ? (d_h = d_ws.h) : (d_h = d_ws.h_guess);

		if ( x < dims.n && y < dims.m )
			d_ws.h_guess[index] = d_h[index] + d_ws.v[index_non_padded];
	}

	template <typename DATATYPE> __global__ void update_solution(reduced_device_workspace<DATATYPE> d_ws, dimensions dims)
	{

		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;
		int index = (y + dims.padding)*dims.n_pad + x + dims.padding;

		if ( x < dims.n && y < dims.m )
			d_ws.h[index] = d_ws.h_guess[index];
	}

	template <typename DATATYPE> __host__ void check_flag_blocks(unified_work_space<DATATYPE> u_ws, dimensions dims, newton_status::status &step_status)
	{

		memory_manager::copyDeviceToHost(u_ws.solution_flags);

		char *h_solution_flags;
		h_solution_flags = u_ws.solution_flags->data_host;


		for (int j = 0; j < dims.m_reduced; j++)
		{
			for (int i = 0; i < dims.n_reduced; i++)
			{
				int k = j*dims.n_reduced + i;
				int status = h_solution_flags[k];

				if (status != newton_status::SUCCESS)
				{
					step_status = (newton_status::status)status;
					break;
				}

			}

			if (step_status != newton_status::SUCCESS) break;
		}
	}


	

	template <typename DATATYPE, model::id MODEL_ID,
		boundary_condtion_type BC_X0, boundary_condtion_type BC_XN,
		boundary_condtion_type BC_Y0, boundary_condtion_type BC_YM,
		bool FIRST_NEWTON_ITERATION > __global__
		void preprocessing(reduced_device_workspace<DATATYPE> d_ws, dimensions dims, spatial_parameters<DATATYPE> spatialParas, model_parameters<DATATYPE, MODEL_ID> modelParas, DATATYPE dt, newton_parameters<DATATYPE> newt_paras)
	{

			int x = blockIdx.x*blockDim.x + threadIdx.x;
			int y = blockIdx.y*blockDim.y + threadIdx.y;

			if (x < dims.n && y < dims.m)
			{

				int index = (y + dims.padding)*dims.n_pad + x +  dims.padding;

				DATATYPE *d_h;
				(FIRST_NEWTON_ITERATION) ? (d_h = d_ws.h) : (d_h = d_ws.h_guess);

				__syncthreads();
	
				compute_nonlinear_functions<DATATYPE,MODEL_ID>(d_ws, d_h, spatialParas, modelParas, dt, index);

				__syncthreads();
				boundary_condition::rhs_vector_Y0_column<DATATYPE, BC_Y0>(d_ws, dims, d_h, x, y, index);

				__syncthreads();
				boundary_condition::rhs_vector_YM_column<DATATYPE, BC_YM>(d_ws, dims, d_h, x, y, index);

				__syncthreads();
				boundary_condition::rhs_vector_X0_row<DATATYPE, BC_X0>(d_ws, dims, d_h, x, y, index);

				__syncthreads();
				boundary_condition::rhs_vector_XN_row<DATATYPE, BC_XN>(d_ws, dims, d_h, x, y, index);

				__syncthreads();
				boundary_condition::rhs_vector_corner_blocks<DATATYPE, BC_X0, BC_XN, BC_Y0, BC_YM>(d_ws, dims, d_h, x, y, index);

			}

		}



	template<typename DATATYPE, model::id MODEL_ID, initial_condition::id IC_ID,
		boundary_condtion_type BC_X0, boundary_condtion_type BC_XN,
		boundary_condtion_type BC_Y0, boundary_condtion_type BC_YM, newton_stage::stage NEWTON_STAGE > void
		apply_single_iteration
		(DATATYPE &dt, unified_work_space<DATATYPE> u_ws, dimensions dims, parameters<DATATYPE,MODEL_ID,IC_ID> paras, cuda_parameters::kernal_launch_parameters ker_launch_paras , newton_status::status &n_status){

			dim3 block_size = ker_launch_paras.simple_sqaure_block.block;
			dim3 thread_size = ker_launch_paras.simple_sqaure_block.thread;
			
			bool const FIRST_NEWTON_ITERATION = ( NEWTON_STAGE == newton_stage::INITIAL );

			preprocessing<DATATYPE, MODEL_ID, BC_X0, BC_XN, BC_Y0, BC_YM, FIRST_NEWTON_ITERATION> << <block_size, thread_size >> >(u_ws.reduced_dev_ws, dims, paras.spatial , paras.model, dt, paras.newton );
			cudaDeviceSynchronize();

			// Apply ADI method in y direction

			block_size = ker_launch_paras.compute_Jy.block;
			thread_size = ker_launch_paras.compute_Jy.thread;

			compute_Jy_and_F<DATATYPE, cuda_parameters::SOLVE_JY_SUBLOOP_SIZE, cuda_parameters::SOLVE_JY_THREAD_SIZE, FIRST_NEWTON_ITERATION, BC_Y0, BC_YM> << <block_size, thread_size >> >(u_ws.reduced_dev_ws, dims);
			cudaDeviceSynchronize();
			thread_size = ker_launch_paras.penta_y_direction.thread;
			block_size = ker_launch_paras.penta_y_direction.block;

			bool const Y_DIRECTION = true;
			
			nonlinear_penta_solver::lu_decomposition<DATATYPE, cuda_parameters::PENTA_LU_LINE_THREAD_SIZE, Y_DIRECTION> << <block_size, thread_size >> >(u_ws.reduced_dev_ws, dims);
			cudaDeviceSynchronize();
			block_size = ker_launch_paras.simple_sqaure_block.block;
			thread_size = ker_launch_paras.simple_sqaure_block.thread;

			// Apply ADI method in x direction

			compute_Jx<DATATYPE, cuda_parameters::SIMPLE_SQUARE_BLOCK, FIRST_NEWTON_ITERATION,BC_X0,BC_XN> << <block_size, thread_size >> >(u_ws.reduced_dev_ws, dims);
			cudaDeviceSynchronize();
			thread_size = ker_launch_paras.penta_x_direction.thread;
			block_size = ker_launch_paras.penta_x_direction.block;

			bool const X_DIRECTION = !(Y_DIRECTION);
			nonlinear_penta_solver::lu_decomposition<DATATYPE, cuda_parameters::PENTA_LU_LINE_THREAD_SIZE, X_DIRECTION> << <block_size, thread_size >> >(u_ws.reduced_dev_ws, dims );
			cudaDeviceSynchronize();

				
			block_size = ker_launch_paras.simple_sqaure_block.block;
			thread_size = ker_launch_paras.simple_sqaure_block.thread;

				n_status = newton_status::SUCCESS;
			if( NEWTON_STAGE == newton_stage::LOOP )
			{
				n_status = newton_status::SUCCESS;
				check_blocks<DATATYPE,FIRST_NEWTON_ITERATION> << <block_size, thread_size >> >( u_ws.reduced_dev_ws , dims , paras.newton );
				check_flag_blocks<DATATYPE>(u_ws, dims, n_status);
			}
			else
				update_guess<DATATYPE,FIRST_NEWTON_ITERATION> << <block_size, thread_size >> >(u_ws.reduced_dev_ws, dims);
			


		}
	
	template<typename DATATYPE, model::id MODEL_ID, initial_condition::id IC_ID, 
		boundary_condtion_type BC_X0, boundary_condtion_type BC_XN,
		boundary_condtion_type BC_Y0, boundary_condtion_type BC_YM> void
			solve_time_step
			(unified_work_space<DATATYPE> u_ws, dimensions dims,
			 parameters<DATATYPE,MODEL_ID,IC_ID> paras, cuda_parameters::kernal_launch_parameters ker_launch_paras , DATATYPE dt , size_t &newton_count , newton_status::status &n_status )
	{

		
		// applying first iteration
		apply_single_iteration<DATATYPE, MODEL_ID, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM, newton_stage::INITIAL >(dt, u_ws, dims, paras, ker_launch_paras, n_status);

		// applying min_iterations without any convergence checks 
		for (newton_count = 2; newton_count < paras.newton.min_iterations ; newton_count++)
		
			apply_single_iteration<DATATYPE, MODEL_ID, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM, newton_stage::PRELOOP>(dt, u_ws, dims, paras, ker_launch_paras , n_status);

		
		// applying remaining iterations with convergence checks 
		for (newton_count = paras.newton.min_iterations + 1 ; newton_count < paras.newton.max_iterations ; newton_count++)
		{
			apply_single_iteration<DATATYPE, MODEL_ID, IC_ID, BC_X0, BC_XN, BC_Y0, BC_YM, newton_stage::LOOP>(dt, u_ws, dims, paras, ker_launch_paras , n_status);
			if (n_status == newton_status::SUCCESS) break;
		}

		// updating solution if successful
		if ( n_status == newton_status::SUCCESS )
		{
				dim3 block_size = ker_launch_paras.simple_sqaure_block.block;
				dim3 thread_size = ker_launch_paras.simple_sqaure_block.thread;
				newton_iterative_method::update_solution<DATATYPE> << < block_size, thread_size >> >(u_ws.reduced_dev_ws, dims);
		}


	}

}




#endif