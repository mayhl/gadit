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
// Name:		compute_Jy_F.h
// Version: 	1.0
// Purpose:		Forms system of penta-diagonal systems solving ADI method in y
//				direction. Saves transpose of data to allow memory coalescence for 
//				ADI method in x direction. 
// CUDA Info:	Solves problem by dividing physical domain into 2D blocks and 
//				assigns an equal size block of threads.   
// ----------------------------------------------------------------------------------



#ifndef LST_TYPE_1_JY
#define LST_TYPE_1_JY


#include "cuda_runtime.h"
#include "work_space.h"
#include "boundary_conditions.h"

namespace compute_Jy_F
{ 

	int const PADDING2 = 2;

	struct indices_Jy
	{
		int globalIdx;
		int shifted_padded_globalIdx_kpp;
		int y;
	};

// loading additional rows 
template <typename DATATYPE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE> __device__
	void load_first_solution_row(
		DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
		DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
		DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
		DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
		DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE],
		indices_Jy &idx,
		reduced_device_workspace<DATATYPE> d_ws, dimensions dims)
{

	DATATYPE *d_h;
	(FIRST_NEWTON_ITERATION) ? (d_h = d_ws.h) : (d_h = d_ws.h_guess);

	int shift_padded_global_kmm = idx.shifted_padded_globalIdx_kpp - 4 * dims.n_pad;
	int shift_padded_global_km = idx.shifted_padded_globalIdx_kpp - 3 * dims.n_pad;
	int shift_padded_global_k0 = idx.shifted_padded_globalIdx_kpp - 2 * dims.n_pad;
	int shift_padded_global_kp = idx.shifted_padded_globalIdx_kpp - 1 * dims.n_pad;

	int load_size;

	(blockIdx.x < gridDim.x - 1) ? load_size = THREAD_SIZE : load_size = dims.Jy_F_last_x_block_size;

	if (load_size >  2 * PADDING2)
	{
		s_h_lmm[threadIdx.x] = d_h[shift_padded_global_kmm];
		s_h_lm[threadIdx.x] = d_h[shift_padded_global_km];
		s_h_l0[threadIdx.x] = d_h[shift_padded_global_k0];
		s_h_lp[threadIdx.x] = d_h[shift_padded_global_kp];

		s_f1_lm[threadIdx.x] = d_ws.f1[shift_padded_global_km];
		s_f1_l0[threadIdx.x] = d_ws.f1[shift_padded_global_k0];

		s_df1_lm[threadIdx.x] = d_ws.df1[shift_padded_global_km];
		s_df1_l0[threadIdx.x] = d_ws.df1[shift_padded_global_k0];

		s_f2_lm[threadIdx.x] = d_ws.f2[shift_padded_global_km];
		s_f2_l0[threadIdx.x] = d_ws.f2[shift_padded_global_k0];

		s_df2_lm[threadIdx.x] = d_ws.df2[shift_padded_global_km];
		s_df2_l0[threadIdx.x] = d_ws.df2[shift_padded_global_k0];

		if (threadIdx.x  < 2 * PADDING2){

			int shift2_threadIdx = threadIdx.x + load_size;

			s_h_lmm[shift2_threadIdx] = d_h[shift_padded_global_kmm + load_size];
			s_h_lm[shift2_threadIdx] = d_h[shift_padded_global_km + load_size];
			s_h_l0[shift2_threadIdx] = d_h[shift_padded_global_k0 + load_size];
			s_h_lp[shift2_threadIdx] = d_h[shift_padded_global_kp + load_size];

			s_f1_lm[shift2_threadIdx] = d_ws.f1[shift_padded_global_km + load_size];
			s_f1_l0[shift2_threadIdx] = d_ws.f1[shift_padded_global_k0 + load_size];

			s_df1_lm[shift2_threadIdx] = d_ws.df1[shift_padded_global_km + load_size];
			s_df1_l0[shift2_threadIdx] = d_ws.df1[shift_padded_global_k0 + load_size];

			s_f2_lm[shift2_threadIdx] = d_ws.f2[shift_padded_global_km + load_size];
			s_f2_l0[shift2_threadIdx] = d_ws.f2[shift_padded_global_k0 + load_size];

			s_df2_lm[shift2_threadIdx] = d_ws.df2[shift_padded_global_km + load_size];
			s_df2_l0[shift2_threadIdx] = d_ws.df2[shift_padded_global_k0 + load_size];

		}
	}
	else
	{
		for (int i = 0; i < load_size + 2 * PADDING2; i += load_size)
		{

			int shift2_threadIdx = threadIdx.x + i;
			s_h_lmm[shift2_threadIdx] = d_h[shift_padded_global_kmm + i];
			s_h_lm[shift2_threadIdx] = d_h[shift_padded_global_km + i];
			s_h_l0[shift2_threadIdx] = d_h[shift_padded_global_k0 + i];
			s_h_lp[shift2_threadIdx] = d_h[shift_padded_global_kp + i];

			s_f1_lm[shift2_threadIdx] = d_ws.f1[shift_padded_global_km + i];
			s_f1_l0[shift2_threadIdx] = d_ws.f1[shift_padded_global_k0 + i];

			s_df1_lm[shift2_threadIdx] = d_ws.df1[shift_padded_global_km + i];
			s_df1_l0[shift2_threadIdx] = d_ws.df1[shift_padded_global_k0 + i];

			s_f2_lm[shift2_threadIdx] = d_ws.f2[shift_padded_global_km + i];
			s_f2_l0[shift2_threadIdx] = d_ws.f2[shift_padded_global_k0 + i];

			s_df2_lm[shift2_threadIdx] = d_ws.df2[shift_padded_global_km + i];
			s_df2_l0[shift2_threadIdx] = d_ws.df2[shift_padded_global_k0 + i];

		}
	}

}

template <typename DATATYPE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE> __device__
	void load_next_solution_row(
		DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
		DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
		DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
		DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
		DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE],
		indices_Jy &idx,
		reduced_device_workspace<DATATYPE> d_ws, dimensions dims)

{

		int shifted_padded_global_kp = idx.shifted_padded_globalIdx_kpp - dims.n_pad;

		DATATYPE *d_h;

		(FIRST_NEWTON_ITERATION) ? (d_h = d_ws.h) : (d_h = d_ws.h_guess);

		int load_size;

		(blockIdx.x < gridDim.x - 1) ? load_size = THREAD_SIZE : load_size = dims.Jy_F_last_x_block_size;

		if (load_size > 2 * PADDING2)
		{
			s_h_lpp[threadIdx.x] = d_h[idx.shifted_padded_globalIdx_kpp];

			s_f1_lp[threadIdx.x] = d_ws.f1[shifted_padded_global_kp];
			s_df1_lp[threadIdx.x] = d_ws.df1[shifted_padded_global_kp];

			s_f2_lp[threadIdx.x] = d_ws.f2[shifted_padded_global_kp];
			s_df2_lp[threadIdx.x] = d_ws.df2[shifted_padded_global_kp];

			if (threadIdx.x < 2 * PADDING2) {

				int shifted_thread_threadIdx = threadIdx.x + load_size;
				int shifted_thread_shifted_padded_global_kpdded_global_kp = shifted_padded_global_kp + load_size;

				s_h_lpp[shifted_thread_threadIdx] = d_h[idx.shifted_padded_globalIdx_kpp + load_size];

				s_f1_lp[shifted_thread_threadIdx] = d_ws.f1[shifted_thread_shifted_padded_global_kpdded_global_kp];
				s_df1_lp[shifted_thread_threadIdx] = d_ws.df1[shifted_thread_shifted_padded_global_kpdded_global_kp];

				s_f2_lp[shifted_thread_threadIdx] = d_ws.f2[shifted_thread_shifted_padded_global_kpdded_global_kp];
				s_df2_lp[shifted_thread_threadIdx] = d_ws.df2[shifted_thread_shifted_padded_global_kpdded_global_kp];

			}

		}
		else
		{
			for (int i = 0; i < load_size + 2 * PADDING2; i += load_size)
			{

				int shifted_thread_threadIdx = threadIdx.x + i;
				int shifted_thread_shifted_padded_global_kpdded_global_lp = shifted_padded_global_kp + i;

				s_h_lpp[shifted_thread_threadIdx] = d_h[idx.shifted_padded_globalIdx_kpp + i];

				s_f1_lp[shifted_thread_threadIdx] = d_ws.f1[shifted_thread_shifted_padded_global_kpdded_global_lp];
				s_df1_lp[shifted_thread_threadIdx] = d_ws.df1[shifted_thread_shifted_padded_global_kpdded_global_lp];

				s_f2_lp[shifted_thread_threadIdx] = d_ws.f2[shifted_thread_shifted_padded_global_kpdded_global_lp];
				s_df2_lp[shifted_thread_threadIdx] = d_ws.df2[shifted_thread_shifted_padded_global_kpdded_global_lp];
			}
		}


}

template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE, boundary_condtion_type::IDs  BC_Y0, boundary_condtion_type::IDs  BC_YM> __device__
	void compute_Jy_and_F_row_single(
		DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
		DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
		DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
		DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
		DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE],
		indices_Jy &idx, reduced_device_workspace<DATATYPE> d_ws, dimensions dims)
{

	load_next_solution_row<DATATYPE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);

	
	int kmm = threadIdx.x;
	int km = threadIdx.x + 1;
	int k0 = threadIdx.x + 2;
	int kp = threadIdx.x + 3;;
	int kpp = threadIdx.x + 4;

	DATATYPE r_nonLinearRHS_divgradlaplace =
		(s_f1_l0[k0] + s_f1_l0[km])*(s_h_l0[kmm] + s_h_lm[km] - 5.0*s_h_l0[km] + s_h_lp[km] - s_h_lm[k0] + 5.0*s_h_l0[k0] - s_h_lp[k0] - s_h_l0[kp]) +
		(s_f1_l0[k0] + s_f1_lm[k0])*(s_h_lmm[k0] + s_h_lm[km] - 5.0*s_h_lm[k0] + s_h_lm[kp] - s_h_l0[km] + 5.0*s_h_l0[k0] - s_h_l0[kp] - s_h_lp[k0]) +
		(s_f1_l0[kp] + s_f1_l0[k0])*(-s_h_l0[km] - s_h_lm[k0] + 5.0*s_h_l0[k0] - s_h_lp[k0] + s_h_lm[kp] - 5.0*s_h_l0[kp] + s_h_lp[kp] + s_h_l0[kpp]) +
		(s_f1_lp[k0] + s_f1_l0[k0])*(-s_h_lm[k0] - s_h_l0[km] + 5.0*s_h_l0[k0] - s_h_l0[kp] + s_h_lp[km] - 5.0*s_h_lp[k0] + s_h_lp[kp] + s_h_lpp[k0]);

	DATATYPE r_nonLinearRHS_divgrad =
		(s_f2_l0[k0] + s_f2_l0[km])*(s_h_l0[km] - s_h_l0[k0]) +
		(s_f2_l0[k0] + s_f2_lm[k0])*(s_h_lm[k0] - s_h_l0[k0]) +
		(s_f2_l0[kp] + s_f2_l0[k0])*(s_h_l0[kp] - s_h_l0[k0]) +
		(s_f2_lp[k0] + s_f2_l0[k0])*(s_h_lp[k0] - s_h_l0[k0]);

	
	DATATYPE r_nonLinearRHS;
	r_nonLinearRHS = r_nonLinearRHS_divgradlaplace + r_nonLinearRHS_divgrad;

	if (FIRST_NEWTON_ITERATION)
	{
		DATATYPE F;
		DATATYPE F_fixed;

		F = -2.0*r_nonLinearRHS;
		F_fixed = s_h_l0[k0] - r_nonLinearRHS;

		d_ws.F_fixed[idx.globalIdx] = F_fixed;
		d_ws.F[idx.globalIdx] = F;

	}
	else
	{
		DATATYPE F_fixed;
		DATATYPE F_guess;

		F_fixed = d_ws.F_fixed[idx.globalIdx];
		F_guess = -s_h_l0[k0] - r_nonLinearRHS + F_fixed;

		d_ws.F[idx.globalIdx] = F_guess;
	}


	DATATYPE  HYYY_LP_L0;		DATATYPE  HYYY_LP_LP;
	DATATYPE  HYYY_L0_L0;		DATATYPE  HYYY_L0_LP;
	DATATYPE  HYYY_LM_L0;		DATATYPE  HYYY_LM_LP;
	DATATYPE  HYYY_LMM_L0;		DATATYPE  HYYY_LMM_LP;

	DATATYPE  HY_L0_L0;			DATATYPE  HY_L0_LP;
	DATATYPE  HY_LM_L0;			DATATYPE  HY_LM_LP;






	penta_diag_row<DATATYPE> Jy;
	bool custom_Jy_row;



	if (2 < idx.y && idx.y < dims.m - 3) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::INTERIOR, BC_Y0, BC_YM>(HYYY_LP_L0, HYYY_LP_LP, HYYY_L0_L0, HYYY_L0_LP, HYYY_LM_L0, HYYY_LM_LP, HYYY_LMM_L0, HYYY_LMM_LP, HY_L0_L0, HY_L0_LP, HY_LM_L0, HY_LM_LP, Jy, custom_Jy_row);

	if (idx.y == 0) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::FIRST, BC_Y0, BC_YM>(HYYY_LP_L0, HYYY_LP_LP, HYYY_L0_L0, HYYY_L0_LP, HYYY_LM_L0, HYYY_LM_LP, HYYY_LMM_L0, HYYY_LMM_LP, HY_L0_L0, HY_L0_LP, HY_LM_L0, HY_LM_LP, Jy, custom_Jy_row);
	if (idx.y == 1) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::SECOND, BC_Y0, BC_YM>(HYYY_LP_L0, HYYY_LP_LP, HYYY_L0_L0, HYYY_L0_LP, HYYY_LM_L0, HYYY_LM_LP, HYYY_LMM_L0, HYYY_LMM_LP, HY_L0_L0, HY_L0_LP, HY_LM_L0, HY_LM_LP, Jy, custom_Jy_row);
	if (idx.y == 2) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::THIRD, BC_Y0, BC_YM>(HYYY_LP_L0, HYYY_LP_LP, HYYY_L0_L0, HYYY_L0_LP, HYYY_LM_L0, HYYY_LM_LP, HYYY_LMM_L0, HYYY_LMM_LP, HY_L0_L0, HY_L0_LP, HY_LM_L0, HY_LM_LP, Jy, custom_Jy_row);

	if (idx.y == (dims.m - 1)) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::FIRST_LAST, BC_Y0, BC_YM>(HYYY_LP_L0, HYYY_LP_LP, HYYY_L0_L0, HYYY_L0_LP, HYYY_LM_L0, HYYY_LM_LP, HYYY_LMM_L0, HYYY_LMM_LP, HY_L0_L0, HY_L0_LP, HY_LM_L0, HY_LM_LP, Jy, custom_Jy_row);
	if (idx.y == (dims.m - 2)) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::SECOND_LAST, BC_Y0, BC_YM>(HYYY_LP_L0, HYYY_LP_LP, HYYY_L0_L0, HYYY_L0_LP, HYYY_LM_L0, HYYY_LM_LP, HYYY_LMM_L0, HYYY_LMM_LP, HY_L0_L0, HY_L0_LP, HY_LM_L0, HY_LM_LP, Jy, custom_Jy_row);
	if (idx.y == (dims.m - 3)) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::THIRD_LAST, BC_Y0, BC_YM>(HYYY_LP_L0, HYYY_LP_LP, HYYY_L0_L0, HYYY_L0_LP, HYYY_LM_L0, HYYY_LM_LP, HYYY_LMM_L0, HYYY_LMM_LP, HY_L0_L0, HY_L0_LP, HY_LM_L0, HY_LM_LP, Jy, custom_Jy_row);

	if (!custom_Jy_row)
	{
		DATATYPE r_f1_lp_half = s_f1_lp[k0] + s_f1_l0[k0];
		DATATYPE r_f1_l0_half = s_f1_l0[k0] + s_f1_lm[k0];

		DATATYPE r_f2_lp_half = s_f2_lp[k0] + s_f2_l0[k0];
		DATATYPE r_f2_l0_half = s_f2_l0[k0] + s_f2_lm[k0];

		DATATYPE r_hy_lp_half = (s_h_lp[k0] - s_h_l0[k0]);
		DATATYPE r_hy_l0_half = (s_h_l0[k0] - s_h_lm[k0]);

		DATATYPE r_hyyy_lp_half = (s_h_lpp[k0] - 3.0* r_hy_lp_half - s_h_lm[k0]);
		DATATYPE r_hyyy_l0_half = (s_h_lp[k0] - 3.0* r_hy_l0_half - s_h_lmm[k0]);

		Jy.c = s_df1_l0[k0] * r_hyyy_lp_half + HYYY_LM_LP*r_f1_lp_half - s_df1_l0[k0] * r_hyyy_l0_half - HYYY_L0_L0*r_f1_l0_half
			 + s_df2_l0[k0] * r_hy_lp_half   +   HY_LM_LP*r_f2_lp_half - s_df2_l0[k0] *   r_hy_l0_half -   HY_L0_L0*r_f2_l0_half;

		Jy.b = -s_df1_lm[k0] * r_hyyy_l0_half - HYYY_LM_L0*r_f1_l0_half + HYYY_LMM_LP*r_f1_lp_half
			   -s_df2_lm[k0] * r_hy_l0_half   -   HY_LM_L0*r_f2_l0_half;


		Jy.d =  s_df1_lp[k0] * r_hyyy_lp_half + HYYY_L0_LP*r_f1_lp_half - HYYY_LP_L0*r_f1_l0_half
			   +s_df2_lp[k0] * r_hy_lp_half   +   HY_L0_LP*r_f2_lp_half;

		Jy.e = r_f1_lp_half*HYYY_LP_LP;
		Jy.a = -r_f1_l0_half*HYYY_LMM_L0;

	}
	else
		d_ws.F[idx.globalIdx] = Jy.f;

	d_ws.Jy_a[idx.globalIdx] = Jy.a;
	d_ws.Jy_b[idx.globalIdx] = Jy.b;
	d_ws.Jy_c[idx.globalIdx] = 1.0 + Jy.c;
	d_ws.Jy_d[idx.globalIdx] = Jy.d;
	d_ws.Jy_e[idx.globalIdx] = Jy.e;


	idx.shifted_padded_globalIdx_kpp += dims.n_pad;
	idx.globalIdx += dims.n;
	idx.y += 1;
	
	
}

template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE,boundary_condtion_type::IDs  BC_Y0, boundary_condtion_type::IDs  BC_YM> __device__
	void compute_Jy_and_F_row_subloop(
	reduced_device_workspace<DATATYPE> d_ws, indices_Jy &idx, dimensions dims,
		DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
		DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
		DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
		DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
		DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE])
{
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
}

template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE, boundary_condtion_type::IDs  BC_Y0, boundary_condtion_type::IDs  BC_YM> __device__
	void compute_Jy_and_F_row_last_subloop(
		reduced_device_workspace<DATATYPE> d_ws, indices_Jy &idx, dimensions dims,
		DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
		DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
		DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
		DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
		DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE] ,int sub_loop_length)
{
		if (sub_loop_length > 0) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		if (sub_loop_length > 1) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		if (sub_loop_length > 2) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		if (sub_loop_length > 3) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		if (sub_loop_length > 4) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		if (sub_loop_length > 5) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		if (sub_loop_length > 6) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		if (sub_loop_length > 7) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		if (sub_loop_length > 8) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		if (sub_loop_length > 9) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		if (sub_loop_length > 10) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		if (sub_loop_length > 11) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		if (sub_loop_length > 12) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		if (sub_loop_length > 13) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		if (sub_loop_length > 14) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
}

template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION,boundary_condtion_type::IDs  BC_Y0, boundary_condtion_type::IDs  BC_YM> __global__
void compute_Jy_and_F
(reduced_device_workspace<DATATYPE> d_ws, dimensions dims)
{

	int const LOAD_SIZE = THREAD_SIZE + 2 * PADDING2;

	__shared__ DATATYPE s_h_lmm[LOAD_SIZE];
	__shared__ DATATYPE s_h_lm[LOAD_SIZE];
	__shared__ DATATYPE s_h_l0[LOAD_SIZE];
	__shared__ DATATYPE s_h_lp[LOAD_SIZE];
	__shared__ DATATYPE s_h_lpp[LOAD_SIZE];

	__shared__ DATATYPE s_f1_lm[LOAD_SIZE];
	__shared__ DATATYPE s_f1_l0[LOAD_SIZE];
	__shared__ DATATYPE s_f1_lp[LOAD_SIZE];

	__shared__ DATATYPE s_f2_lm[LOAD_SIZE];
	__shared__ DATATYPE s_f2_l0[LOAD_SIZE];
	__shared__ DATATYPE s_f2_lp[LOAD_SIZE];

	__shared__ DATATYPE s_df1_lm[LOAD_SIZE];
	__shared__ DATATYPE s_df1_l0[LOAD_SIZE];
	__shared__ DATATYPE s_df1_lp[LOAD_SIZE];

	__shared__ DATATYPE s_df2_lm[LOAD_SIZE];
	__shared__ DATATYPE s_df2_l0[LOAD_SIZE];
	__shared__ DATATYPE s_df2_lp[LOAD_SIZE];


	int x = blockIdx.x*THREAD_SIZE + threadIdx.x;
	int y = blockIdx.y*BLOCK_SIZE;

	indices_Jy idx;

	idx.shifted_padded_globalIdx_kpp = (y + 2 * PADDING2)*dims.n_pad + x;
	idx.globalIdx = (y)*dims.n + x;
	idx.y = y;

	//idx.valid_x_point = (x < dims.n);
	if (x < dims.n)
	{
		load_first_solution_row<DATATYPE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);

		if (blockIdx.y < gridDim.y - 1)
			compute_Jy_and_F_row_subloop<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(
			d_ws, idx, dims,
			s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp,
			s_f1_lm, s_f1_l0, s_f1_lp,
			s_df1_lm, s_df1_l0, s_df1_lp,
			s_f2_lm, s_f2_l0, s_f2_lp,
			s_df2_lm, s_df2_l0, s_df2_lp);
		else
			compute_Jy_and_F_row_last_subloop<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(
			d_ws, idx, dims,
			s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp,
			s_f1_lm, s_f1_l0, s_f1_lp,
			s_df1_lm, s_df1_l0, s_df1_lp,
			s_f2_lm, s_f2_l0, s_f2_lp,
			s_df2_lm, s_df2_l0, s_df2_lp,
			dims.Jy_F_last_y_block_size);

	}
}

}
#endif