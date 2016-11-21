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
// Name:		nonlinear_pental_solver.h
// Version: 	1.0
// Purpose:		Solves system of penta-diagonal matrices.
// CUDA Info:	Solves problem by dividing physical domain into  stripes in direction
//				of ADI step and assigns a vector of threads to solve stripe. 
//				transpose of solution to facilitate memory coalescence for solver in
//				transpose direction.
// ----------------------------------------------------------------------------------

#ifndef NONLINEAR_PENTA_SOLVER
#define NONLINEAR_PENTA_SOLVER

#include "work_space.h"

namespace nonlinear_penta_solver
{
	int const INITIAL_STEP_SIZE = 2;
	int const DOWN_SOLVE_SUB_LOOP = 3;
	int const UP_SOLVE_SUB_LOOP = 2;

	int const SOLUTION_CHECK_FLAG_COUNT = 2;



	template <typename DATATYPE> struct penta_workspace
	{


		DATATYPE *LU_a;
		DATATYPE *LU_b;
		DATATYPE *LU_c;
		DATATYPE *LU_d;
		DATATYPE *LU_e;

		DATATYPE *x;
		DATATYPE *y_transpose;
		DATATYPE *f;
	};



	// routines to transfer penta diagonal matrix row data between global memory and register memory
#pragma region row_data_exchange_global_shared
	template <typename DATATYPE> __device__ __forceinline__ void
		load_row_all
		(penta_workspace<DATATYPE> pent_ws, penta_diag_row<DATATYPE> &A, int  globalIdx){

			A.a = pent_ws.LU_a[globalIdx];
			A.b = pent_ws.LU_b[globalIdx];
			A.c = pent_ws.LU_c[globalIdx];
			A.d = pent_ws.LU_d[globalIdx];
			A.e = pent_ws.LU_e[globalIdx];

			A.f = pent_ws.f[globalIdx];

		}

	template <typename DATATYPE> __device__ __forceinline__ void
		load_row_up_step
		(penta_workspace<DATATYPE> pent_ws, penta_diag_row<DATATYPE> &LU, int  globalIdx){


			LU.c = pent_ws.LU_c[globalIdx];
			LU.d = pent_ws.LU_d[globalIdx];
			LU.e = pent_ws.LU_e[globalIdx];
			LU.f = pent_ws.x[globalIdx];


		}

	template <typename DATATYPE> __device__ __forceinline__ void
		save_row_all
		(penta_workspace<DATATYPE> pent_ws, penta_diag_row<DATATYPE> &A, int globalIdx){


			pent_ws.LU_a[globalIdx] = A.a;
			pent_ws.LU_b[globalIdx] = A.b;
			pent_ws.LU_c[globalIdx] = A.c;
			pent_ws.LU_d[globalIdx] = A.d;
			pent_ws.LU_e[globalIdx] = A.e;

			pent_ws.f[globalIdx] = A.f;

		}

	template <typename DATATYPE, bool DEBUG_MODE> __device__ __forceinline__ void
		save_row_down_step
		(penta_workspace<DATATYPE> pent_ws, penta_diag_row<DATATYPE> &LU, int globalIdx)
	{

			if (DEBUG_MODE)
			{
				save_row_all(pent_ws, LU, globalIdx);
			}
			else
			{
				pent_ws.LU_c[globalIdx] = LU.c;
				pent_ws.LU_d[globalIdx] = LU.d;
				pent_ws.LU_e[globalIdx] = LU.e;
				pent_ws.x[globalIdx] = LU.f;
			}

		}


#pragma endregion row_data_exchange_global_register


	// routines used on down solve i.e. solve Lx=f;
#pragma region down_solve

	template <typename DATATYPE, bool DEBUG_MODE> __device__ __forceinline__ void
		lu_down_solve_initial_step
		(penta_workspace<DATATYPE> pent_ws, int const n, int &globalIdx,
		penta_diag_row<DATATYPE>&LU, penta_diag_row<DATATYPE> &LUm, penta_diag_row<DATATYPE> &LUmm)
	{


			// first step
			load_row_all<DATATYPE>(pent_ws, LUmm, globalIdx);

			LUmm.c = 1.0 / LUmm.c;

			save_row_down_step<DATATYPE, DEBUG_MODE>(pent_ws, LUmm, globalIdx);

			// second step

			globalIdx += n;

			load_row_all<DATATYPE>(pent_ws, LUm, globalIdx);

			LUm.b = LUm.b*LUmm.c;
			LUm.c = 1.0 / (LUm.c - LUm.b*LUmm.d);
			LUm.d = LUm.d - LUm.b*LUmm.e;

			LUm.f = LUm.f - LUm.b*LUmm.f;

			save_row_down_step<DATATYPE, DEBUG_MODE>(pent_ws, LUm, globalIdx);

		}

	template <typename DATATYPE, bool DEBUG_MODE> __device__ __forceinline__ void
		lu_down_solve_next_row
		(penta_workspace<DATATYPE> pent_ws, int const n, int &globalIdx,
		penta_diag_row<DATATYPE>&LU, penta_diag_row<DATATYPE> &LUm, penta_diag_row<DATATYPE> &LUmm)
	{


			penta_diag_row<DATATYPE> A;

			globalIdx += n;

			load_row_all(pent_ws, A, globalIdx);


			LU.a = A.a*LUmm.c;
			LU.b = (A.b - LU.a*LUmm.d)*LUm.c;
			LU.c = 1.0 / (A.c - LU.b*LUm.d - LU.a*LUmm.e);
			LU.d = A.d - LU.b*LUmm.e;
			LU.e = A.e;

			LU.f = A.f - LU.a*LUmm.f - LU.b*LUm.f;

			save_row_down_step<DATATYPE, DEBUG_MODE>(pent_ws, LU, globalIdx);

		}

	template <typename DATATYPE, bool DEBUG_MODE> __device__ __forceinline__ void
		lu_down_solve
		(penta_workspace<DATATYPE> pent_ws, penta_diag_row<DATATYPE>&LU, penta_diag_row<DATATYPE> &LUm, penta_diag_row<DATATYPE> &LUmm,
		int const n, int const m, int &globalIdx)
	{

			// To remove memory swaps in two (2) point iterative method
			// steps are unrolled by three (3) with swapping doing in function calls

			// solving first two (2) rows
			lu_down_solve_initial_step<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LU, LUm, LUmm);

			// solving every three (3) row

			int j;
			for ( j = INITIAL_STEP_SIZE; j < m - DOWN_SOLVE_SUB_LOOP; j += DOWN_SOLVE_SUB_LOOP)
			{
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LU, LUm, LUmm);
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LUmm, LU, LUm);
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LUm, LUmm, LU);
			}

			// solving remaing rows (1 to 3) when (m - 2) [-2 due to lu_down_solve_initial_step]
			// is not divisible by DOWN_SOLVE_SUB_LOOP=3
			int remaining_rows = m - j;

			if (remaining_rows == 3)
			{
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LU, LUm, LUmm);
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LUmm, LU, LUm);
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LUm, LUmm, LU);
			}
			else if (remaining_rows == 1)
			{
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LU, LUm, LUmm);

			}
			else //if down_solve_end_loop_size == 2
			{
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LU, LUm, LUmm);
				lu_down_solve_next_row<DATATYPE, DEBUG_MODE>(pent_ws, n, globalIdx, LUmm, LU, LUm);
			}


		}


#pragma endregion down_solve

	// routines used on up solve i.e. solve Uy=x;
#pragma region up_solve

	template <typename DATATYPE, int THREAD_SIZE> __device__ __forceinline__ void
		lu_up_solve_row_pair
		(penta_workspace<DATATYPE> pent_ws,
		penta_diag_row<DATATYPE>&LU, penta_diag_row<DATATYPE> &LUm, penta_diag_row<DATATYPE> &LUmm, DATATYPE s_y[THREAD_SIZE][THREAD_SIZE + 1],
		int &globalIdx, int &globalIdx_transpose, int const sub_loop_index, int const n)
	{



			globalIdx -= n;

			load_row_up_step<DATATYPE>(pent_ws, LUmm, globalIdx);
			LU.f = (LUmm.f - LUmm.e*LU.f - LUmm.d*LUm.f)*LUmm.c;
			s_y[sub_loop_index][threadIdx.x] = LU.f;


			globalIdx -= n;

			load_row_up_step<DATATYPE>(pent_ws, LUmm, globalIdx);
			LUm.f = (LUmm.f - LUmm.e*LUm.f - LUmm.d*LU.f)*LUmm.c;
			s_y[sub_loop_index + 1][threadIdx.x] = LUm.f;

		}

	template <typename DATATYPE, int THREAD_SIZE> __device__ __forceinline__ void
		lu_up_solve_row_single_last
		(penta_workspace<DATATYPE> pent_ws,
		penta_diag_row<DATATYPE>&LU, penta_diag_row<DATATYPE> &LUm, penta_diag_row<DATATYPE> &LUmm, DATATYPE s_y[THREAD_SIZE][THREAD_SIZE + 1],
		int &globalIdx, int &globalIdx_transpose, int const sub_loop_index, int const n)
	{


			globalIdx -= n;

			load_row_up_step<DATATYPE>(pent_ws, LUmm, globalIdx);
			LU.f = (LUmm.f - LUmm.e*LU.f - LUmm.d*LUm.f)*LUmm.c;
			s_y[sub_loop_index][threadIdx.x] = LU.f;


		}


	template <typename DATATYPE, int THREAD_SIZE, bool TOP_Y_BLOCK> __device__ __forceinline__ void
		save_transpose_block
		(penta_workspace<DATATYPE> p_ws, DATATYPE s_y[THREAD_SIZE][THREAD_SIZE + 1],
		int &globalIdx_transpose, int const m,
		int const up_solve_end_loop_size_x, int const up_solve_end_loop_size_y)
	{

			__syncthreads();

			int tmp_index = globalIdx_transpose;

			int col_size;

			(blockIdx.x < gridDim.x - 1) ? (col_size = THREAD_SIZE) : (col_size = up_solve_end_loop_size_x);

			if ((!(TOP_Y_BLOCK) || threadIdx.x < up_solve_end_loop_size_y))
			{
				for (int j = 0; j < col_size; j++)
				{
					// save y row for fixed x
					p_ws.y_transpose[tmp_index] = s_y[threadIdx.x][j];
					tmp_index += m;
				}
			}

			globalIdx_transpose -= THREAD_SIZE;

		}


	template <typename DATATYPE, int THREAD_SIZE> __device__ __forceinline__ void
		lu_up_solve_last_block
		(penta_workspace<DATATYPE> pent_ws, penta_diag_row<DATATYPE>&LU, penta_diag_row<DATATYPE> &LUm, penta_diag_row<DATATYPE> &LUmm,
		DATATYPE s_y[THREAD_SIZE][THREAD_SIZE + 1], int &globalIdx, int &globalIdx_transpose, int const n)
	{


			// step n
			load_row_up_step<DATATYPE>(pent_ws, LU, globalIdx);

			LU.f = LU.f*LU.c;

			//d_ws.y[globalIdx] = LU.f;
			s_y[0][threadIdx.x] = LU.f;

			//// step n-1
			globalIdx -= n;

			load_row_up_step<DATATYPE>(pent_ws, LUm, globalIdx);

			LUm.f = (LUm.f - LUm.d*LU.f)*LUm.c;
			//d_ws.y[globalIdx] = LUm.f;
			s_y[1][threadIdx.x] = LUm.f;


			// solve next THREAD_SIZE - INITIAL_STEP_SIZE  up steps

			for (int i = INITIAL_STEP_SIZE; i < THREAD_SIZE; i += UP_SOLVE_SUB_LOOP)
				lu_up_solve_row_pair<DATATYPE, THREAD_SIZE>(pent_ws, LU, LUm, LUmm, s_y, globalIdx, globalIdx_transpose, i, n);


		}


	template <typename DATATYPE, int THREAD_SIZE> __device__ __forceinline__ void
		lu_up_solve_remaining_blocks
		(penta_workspace<DATATYPE> pent_ws, penta_diag_row<DATATYPE>&LU, penta_diag_row<DATATYPE> &LUm, penta_diag_row<DATATYPE> &LUmm, DATATYPE s_y[THREAD_SIZE][THREAD_SIZE + 1],
		int &globalIdx, int &globalIdx_transpose,
		int const n, int const m, int const m_pad,
		int const up_solve_end_loop_size_x, int const up_solve_end_loop_size_y, const bool valid_point)
	{
			bool const TOP_Y_BLOCK = true;
			bool const NOT_TOP_Y_BLOCK = !(TOP_Y_BLOCK);

			int j = THREAD_SIZE;

			for (j; j < m - THREAD_SIZE; j += THREAD_SIZE)
			{
				// solve next THREAD_SIZE up steps
				if (valid_point)
				for (int i = 0; i < THREAD_SIZE; i += UP_SOLVE_SUB_LOOP)
					lu_up_solve_row_pair<DATATYPE, THREAD_SIZE>(pent_ws, LU, LUm, LUmm, s_y, globalIdx, globalIdx_transpose, i, n);

				save_transpose_block<DATATYPE, THREAD_SIZE, NOT_TOP_Y_BLOCK>(pent_ws, s_y, globalIdx_transpose, m, up_solve_end_loop_size_x, up_solve_end_loop_size_y);
			}


			if (valid_point) {
				int i = 0;
				for (j; j < m - UP_SOLVE_SUB_LOOP; j += UP_SOLVE_SUB_LOOP)
				{
					lu_up_solve_row_pair<DATATYPE, THREAD_SIZE>(pent_ws, LU, LUm, LUmm, s_y, globalIdx, globalIdx_transpose, i, n);
					i += UP_SOLVE_SUB_LOOP;
				}

				if ((m - j) == UP_SOLVE_SUB_LOOP)
					lu_up_solve_row_pair<DATATYPE, THREAD_SIZE>(pent_ws, LU, LUm, LUmm, s_y, globalIdx, globalIdx_transpose, i, n);
				else
					lu_up_solve_row_single_last<DATATYPE, THREAD_SIZE>(pent_ws, LU, LUm, LUmm, s_y, globalIdx, globalIdx_transpose, i, n);
			}


			save_transpose_block<DATATYPE, THREAD_SIZE, TOP_Y_BLOCK>(pent_ws, s_y, globalIdx_transpose, m, up_solve_end_loop_size_x, up_solve_end_loop_size_y);



		}

#pragma endregion up_solve

	template <typename DATATYPE, int THREAD_SIZE, bool Y_DIRECTION > __global__ void
		lu_decomposition(reduced_device_workspace<DATATYPE> d_ws, dimensions dims )//, newton_status volatile &status)
	{


			//L U factorization

			//Description: 
			//	Solves a system of independent penta diagonal matrix, A x = f,  using LU decomposition.
			//     Down step computes Ly=f and up step computes Ux=y.

			//Notes: 
			//	LU 


			//Values: 
			//	d_ws.J_i (i=a,b,c,d,e)				penta diagonal matrix row element stored in global memory 

			//	Lj.i (i=a,b,c,d,e) (j='',m,mm)		LDU diagonal matrix row element stored in register memory
			//										at: '', current row; m, previous row; mm, row before previous.
			//										Note, Lj.c stores 1/c instead to speed up 



			//set 'DEBUG_MODE' to true to write L matrix back to global memory.

			bool const DEBUG_MODE = false;

			int n;
			int m;
			int m_pad;
			int up_solve_end_loop_size_y;
			int up_solve_end_loop_size_x;

			penta_workspace<DATATYPE> pent_ws;

			// rotating cords for coalescence 
			if (Y_DIRECTION)
			{
				n = dims.n;
				m = dims.m;
				m_pad = dims.m_pad;

				pent_ws.LU_a = d_ws.Jy_a;
				pent_ws.LU_b = d_ws.Jy_b;
				pent_ws.LU_c = d_ws.Jy_c;
				pent_ws.LU_d = d_ws.Jy_d;
				pent_ws.LU_e = d_ws.Jy_e;

				pent_ws.f = d_ws.F;
				pent_ws.x = d_ws.F;
				pent_ws.y_transpose = d_ws.w_transpose;

				up_solve_end_loop_size_y = dims.penta_lu_up_end_loop_size_y;
				up_solve_end_loop_size_x = dims.penta_lu_up_end_loop_size_x;

			}
			else
			{
				n = dims.m;
				m = dims.n;
				m_pad = dims.n_pad;

				pent_ws.LU_a = d_ws.Jx_a;
				pent_ws.LU_b = d_ws.Jx_b;
				pent_ws.LU_c = d_ws.Jx_c;
				pent_ws.LU_d = d_ws.Jx_d;
				pent_ws.LU_e = d_ws.Jx_e;

				pent_ws.f = d_ws.w_transpose;
				pent_ws.x = d_ws.w_transpose; //????
				pent_ws.y_transpose = d_ws.v;


				up_solve_end_loop_size_y = dims.penta_lu_up_end_loop_size_x;
				up_solve_end_loop_size_x = dims.penta_lu_up_end_loop_size_y;

			}

			// flags to call special form of 'save_transpose_block' incase of non-square TOP_Y_BLOCK i.e. m not divisable by THREAD_SIZE
			bool const TOP_Y_BLOCK = true;
			bool const NOT_TOP_Y_BLOCK = !(TOP_Y_BLOCK);

			// Note: 'dims' and 'd_ws' should no longer be used.


			int globalIdx = blockIdx.x*THREAD_SIZE + threadIdx.x;
			int globalIdx_transpose = (blockIdx.x*THREAD_SIZE)*m + m - 1 - threadIdx.x;


			__shared__ DATATYPE s_y[THREAD_SIZE][THREAD_SIZE + 1];
			penta_diag_row<DATATYPE> LUmm, LUm, LU;

			int x = blockIdx.x*THREAD_SIZE + threadIdx.x;

			// Valid point turn off threads in last x block when referencing points outside the x domain
			// All threads are when 'lu_up_solve_last_block<...>(...)' is called to improve data coalescence.
			bool valid_point = (x < n);

			if (valid_point) {

				lu_down_solve<DATATYPE, DEBUG_MODE>(pent_ws, LU, LUm, LUmm, n, m, globalIdx);
				lu_up_solve_last_block<DATATYPE, THREAD_SIZE>(pent_ws, LU, LUm, LUmm, s_y, globalIdx, globalIdx_transpose, n);

			}

			save_transpose_block<DATATYPE, THREAD_SIZE, NOT_TOP_Y_BLOCK>(pent_ws, s_y, globalIdx_transpose, m, up_solve_end_loop_size_x, up_solve_end_loop_size_y);

			lu_up_solve_remaining_blocks<DATATYPE, THREAD_SIZE>
				(pent_ws, LU, LUm, LUmm, s_y,
				globalIdx, globalIdx_transpose, n, m, m_pad,
				up_solve_end_loop_size_x, up_solve_end_loop_size_y, valid_point);

		}

}
#endif