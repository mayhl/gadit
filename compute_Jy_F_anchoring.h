#ifndef LST_TYPE_2_JY
#define LST_TYPE_2_JY


#include "cuda_runtime.h"
#include "work_space.h"

#include "boundary_conditions.h"

namespace compute_Jy_F_anchoring
{

	int const PADDING2 = 2;
	struct indices_Jy
	{
		int globalIdx;
		int shifted_padded_globalIdx_kpp;
		int y;
		//bool valid_x_point;
	};

	template <typename DATATYPE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE> __device__
		void load_first_solution_row(
			DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE],
			DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE],
			DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE],
			DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE],
			DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE],
			DATATYPE s_mu1_lm[LOAD_SIZE], DATATYPE s_mu1_l0[LOAD_SIZE],
			DATATYPE s_mu2_lm[LOAD_SIZE], DATATYPE s_mu2_l0[LOAD_SIZE],
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

		if (load_size > 2 * PADDING2)
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

			s_mu1_lm[threadIdx.x] = d_ws.mu1[shift_padded_global_km];
			s_mu1_l0[threadIdx.x] = d_ws.mu1[shift_padded_global_k0];

			s_mu2_lm[threadIdx.x] = d_ws.mu2[shift_padded_global_km];
			s_mu2_l0[threadIdx.x] = d_ws.mu2[shift_padded_global_k0];

			if (threadIdx.x < 2 * PADDING2) {

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

				s_mu1_lm[shift2_threadIdx] = d_ws.mu1[shift_padded_global_km + load_size];
				s_mu1_l0[shift2_threadIdx] = d_ws.mu1[shift_padded_global_k0 + load_size];

				s_mu2_lm[shift2_threadIdx] = d_ws.mu2[shift_padded_global_km + load_size];
				s_mu2_l0[shift2_threadIdx] = d_ws.mu2[shift_padded_global_k0 + load_size];

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

				s_mu1_lm[shift2_threadIdx] = d_ws.mu1[shift_padded_global_km + i];
				s_mu1_l0[shift2_threadIdx] = d_ws.mu1[shift_padded_global_k0 + i];

				s_mu2_lm[shift2_threadIdx] = d_ws.mu2[shift_padded_global_km + i];
				s_mu2_l0[shift2_threadIdx] = d_ws.mu2[shift_padded_global_k0 + i];

			}
		}

	}

	template <typename DATATYPE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE> __device__
		void load_next_solution_row(
			DATATYPE s_h_lpp[LOAD_SIZE],
			DATATYPE s_f1_lp[LOAD_SIZE],
			DATATYPE s_df1_lp[LOAD_SIZE],
			DATATYPE s_f2_lp[LOAD_SIZE],
			DATATYPE s_df2_lp[LOAD_SIZE],
			DATATYPE s_mu1_lp[LOAD_SIZE],
			DATATYPE s_mu2_lp[LOAD_SIZE],
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

			s_mu1_lp[threadIdx.x] = d_ws.mu1[shifted_padded_global_kp];
			s_mu2_lp[threadIdx.x] = d_ws.mu2[shifted_padded_global_kp];

			if (threadIdx.x < 2 * PADDING2) {

				int shifted_thread_threadIdx = threadIdx.x + load_size;
				int shifted_thread_shifted_padded_global_kpdded_global_kp = shifted_padded_global_kp + load_size;

				s_h_lpp[shifted_thread_threadIdx] = d_h[idx.shifted_padded_globalIdx_kpp + load_size];

				s_f1_lp[shifted_thread_threadIdx] = d_ws.f1[shifted_thread_shifted_padded_global_kpdded_global_kp];
				s_df1_lp[shifted_thread_threadIdx] = d_ws.df1[shifted_thread_shifted_padded_global_kpdded_global_kp];

				s_f2_lp[shifted_thread_threadIdx] = d_ws.f2[shifted_thread_shifted_padded_global_kpdded_global_kp];
				s_df2_lp[shifted_thread_threadIdx] = d_ws.df2[shifted_thread_shifted_padded_global_kpdded_global_kp];

				s_mu1_lp[shifted_thread_threadIdx] = d_ws.mu1[shifted_thread_shifted_padded_global_kpdded_global_kp];
				s_mu2_lp[shifted_thread_threadIdx] = d_ws.mu2[shifted_thread_shifted_padded_global_kpdded_global_kp];

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

				s_mu1_lp[shifted_thread_threadIdx] = d_ws.mu1[shifted_thread_shifted_padded_global_kpdded_global_lp];
				s_mu2_lp[shifted_thread_threadIdx] = d_ws.mu2[shifted_thread_shifted_padded_global_kpdded_global_lp];
			}
		}


	}

	template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE, boundary_condtion_type::IDs BC_Y0, boundary_condtion_type::IDs BC_YM> __device__
		void compute_Jy_and_F_row_single(
			DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
			DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
			DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
			DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
			DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE],
			DATATYPE s_mu1_lm[LOAD_SIZE], DATATYPE s_mu1_l0[LOAD_SIZE], DATATYPE s_mu1_lp[LOAD_SIZE],
			DATATYPE s_mu2_lm[LOAD_SIZE], DATATYPE s_mu2_l0[LOAD_SIZE], DATATYPE s_mu2_lp[LOAD_SIZE],
			indices_Jy &idx, reduced_device_workspace<DATATYPE> d_ws, dimensions dims)
	{

		load_next_solution_row<DATATYPE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE>(s_h_lpp, s_f1_lp, s_df1_lp, s_f2_lp, s_df2_lp, s_mu1_lp, s_mu2_lp, idx, d_ws, dims);


		int kmm = threadIdx.x;
		int km = threadIdx.x + 1;
		int k0 = threadIdx.x + 2;
		int kp = threadIdx.x + 3;;
		int kpp = threadIdx.x + 4;

		//DATATYPE r_nonLinearRHS_divgradlaplace =
		//	(s_f1_l0[k0] + s_f1_l0[km])*(s_h_l0[kmm] + s_h_lm[km] - 5.0*s_h_l0[km] + s_h_lp[km] - s_h_lm[k0] + 5.0*s_h_l0[k0] - s_h_lp[k0] - s_h_l0[kp]) +
		//	(s_f1_l0[k0] + s_f1_lm[k0])*(s_h_lmm[k0] + s_h_lm[km] - 5.0*s_h_lm[k0] + s_h_lm[kp] - s_h_l0[km] + 5.0*s_h_l0[k0] - s_h_l0[kp] - s_h_lp[k0]) +
		//	(s_f1_l0[kp] + s_f1_l0[k0])*(-s_h_l0[km] - s_h_lm[k0] + 5.0*s_h_l0[k0] - s_h_lp[k0] + s_h_lm[kp] - 5.0*s_h_l0[kp] + s_h_lp[kp] + s_h_l0[kpp]) +
		//	(s_f1_lp[k0] + s_f1_l0[k0])*(-s_h_lm[k0] - s_h_l0[km] + 5.0*s_h_l0[k0] - s_h_l0[kp] + s_h_lp[km] - 5.0*s_h_lp[k0] + s_h_lp[kp] + s_h_lpp[k0]);

		//DATATYPE r_nonLinearRHS_divgrad =
		//	(s_f2_l0[k0] + s_f2_l0[km])*(s_h_l0[km] - s_h_l0[k0]) +
		//	(s_f2_l0[k0] + s_f2_lm[k0])*(s_h_lm[k0] - s_h_l0[k0]) +
		//	(s_f2_l0[kp] + s_f2_l0[k0])*(s_h_l0[kp] - s_h_l0[k0]) +
		//	(s_f2_lp[k0] + s_f2_l0[k0])*(s_h_lp[k0] - s_h_l0[k0]);




		/*DATATYPE r_nonLinearRHS_divgradlaplace =
		(s_f1_l0[k0] + s_f1_l0[km])*
		(
		(s_h_l0[kmm] + s_h_lm[km] - 5*s_h_l0[km] + s_h_lp[km] - s_h_lm[k0] + 5*s_h_l0[k0] - s_h_lp[k0] - s_h_l0[kp])*mua(0,0) +
		(s_h_lm[kmm] - 4*s_h_lp[kmm] + s_h_lmm[km] - 2*s_h_lm[km] - s_h_l0[km] + 5*s_h_lp[km] - s_h_lm[k0] + 4*s_h_lp[k0] - 4*s_h_lmm[kp] + 5*s_h_lm[kp] + 4*s_h_l0[kp] - 8*s_h_lp[kp])*mub(0,0)*0.25
		) +
		(s_f1_l0[kp] + s_f1_l0[k0])*
		(
		(-s_h_l0[km] - s_h_lm[k0] + 5*s_h_l0[k0] - s_h_lp[k0] + s_h_lm[kp] - 5*s_h_l0[kp] + s_h_lp[kp] + s_h_l0[kpp])*mua(1,0) +
		(-s_h_lm[km] + 4*s_h_lp[km] - s_h_lmm[k0] + 2*s_h_lm[k0] + s_h_l0[k0] - 5*s_h_lp[k0] + s_h_lm[kp] - 4*s_h_lp[kp] + 4*s_h_lmm[kpp] - 5*s_h_lm[kpp] - 4*s_h_l0[kpp] + 8*s_h_lp[kpp])*mub(1,0)*0.25
		) +
		(s_f1_l0[k0] + s_f1_lm[k0])*
		(
		(s_h_lm[km] - s_h_l0[km] + s_h_lmm[k0] - 5*s_h_lm[k0] + 5*s_h_l0[k0] - s_h_lp[k0] + s_h_lm[kp] - s_h_l0[kp])*mud(0,0) +
		(s_h_lm[kmm] + s_h_l0[kmm] + s_h_lmm[km] - 3*s_h_lm[km] - 3*s_h_l0[km] + s_h_lp[km] - 4*s_h_lmm[kp] + 6*s_h_lm[kp] + 6*s_h_l0[kp] - 4*s_h_lp[kp] - s_h_lm[kpp] - s_h_l0[kpp])*muc(0,0)*0.25 +
		) +
		(s_f1_lp[k0] + s_f1_l0[k0])*
		(
		(-s_h_l0[km] + s_h_lp[km] - s_h_lm[k0] + 5*s_h_l0[k0] - 5*s_h_lp[k0] + s_h_lpp[k0] - s_h_l0[kp] + s_h_lp[kp])*mud(0,1) -
		(s_h_l0[kmm] + s_h_lp[kmm] + s_h_lm[km] - 3*s_h_l0[km] - 3*s_h_lp[km] + s_h_lpp[km] - 4*s_h_lm[kp] + 6*s_h_l0[kp] + 6*s_h_lp[kp] - 4*s_h_lpp[kp] - s_h_l0[kpp] - s_h_lp[kpp])*muc(0,1)*0.25
		);*/

		//DATATYPE r_nonLinearRHS_divgrad =
		//          (s_f2_l0[k0] + s_f2_l0[km])*
		//			(	
		//				(s_h_l0[km] - s_h_l0[k0])*mua(0,0) + 
		//				(s_h_lm[km] - s_h_lp[km] + s_h_lm[k0] - s_h_lp[k0])*mub(0,0)*0.25
		//			) +
		//		    (s_f2_l0[kp] + s_f2_l0[k0])*
		//			(
		//			  (-s_h_l0[k0] + s_h_l0[kp])*mua(1,0) +
		//			  (-s_h_lm[k0] + s_h_lp[k0] - s_h_lm[kp] + s_h_lp[kp])*mub(1,0)*0.25
		//			) +
		//			(s_f1_l0[k0] + s_f1_lm[k0])*
		//			(
		//				(s_h_lm[k0] - s_h_l0[k0])*mud(0,0) 
		//				(s_h_lm[km] + s_h_l0[km] - s_h_lm[kp] - s_h_l0[kp])*muc(0,0)*0.25 
		//			) +			
		//			(s_f1_lp[k0] + s_f1_l0[k0])*
		//			(
		//				(-s_h_l0[k0] + s_h_lp[k0])*mud(0,1)
		//				(-s_h_l0[km] - s_h_lp[km] + s_h_l0[kp] + s_h_lp[kp])*muc(0,1)*0.25
		//			);



		//DATATYPE const lambda = 1.8;
		//DATATYPE const nu = 1 * 1.3;

		DATATYPE const lambda = 0.75;
		DATATYPE const nu = 1.0 - lambda;

		DATATYPE mu11kp = lambda + nu*(s_mu1_l0[kp] + s_mu1_l0[k0]);
		DATATYPE mu11k0 = lambda + nu*(s_mu1_l0[k0] + s_mu1_l0[km]);

		DATATYPE mu12kp = 0.25*nu*(s_mu2_l0[kp] + s_mu2_l0[k0]);
		DATATYPE mu12k0 = 0.25*nu *(s_mu2_l0[k0] + s_mu2_l0[km]);


		DATATYPE mu22lp = lambda - nu*(s_mu1_lp[k0] + s_mu1_l0[k0]);
		DATATYPE mu22l0 = lambda - nu *(s_mu1_l0[k0] + s_mu1_lm[k0]);

		DATATYPE mu21lp = 0.25*nu*(s_mu2_lp[k0] + s_mu2_l0[k0]);
		DATATYPE mu21l0 = 0.25*nu*(s_mu2_l0[k0] + s_mu2_lm[k0]);




		DATATYPE r_nonLinearRHS_divgradlaplace =
			(s_f1_l0[k0] + s_f1_l0[km])*
			(
				mu11k0*(s_h_l0[kmm] + s_h_lm[km] - 5 * s_h_l0[km] + s_h_lp[km] - s_h_lm[k0] + 5 * s_h_l0[k0] - s_h_lp[k0] - s_h_l0[kp]) +
				mu12k0*(s_h_lm[kmm] - s_h_lp[kmm] + s_h_lmm[km] - 3 * s_h_lm[km] + 3 * s_h_lp[km] - s_h_lpp[km] + s_h_lmm[k0] - 3 * s_h_lm[k0] + 3 * s_h_lp[k0] - s_h_lpp[k0] + s_h_lm[kp] - s_h_lp[kp])
				) +
				(s_f1_l0[kp] + s_f1_l0[k0])*
			(
				mu11kp*(-s_h_l0[km] - s_h_lm[k0] + 5 * s_h_l0[k0] - s_h_lp[k0] + s_h_lm[kp] - 5 * s_h_l0[kp] + s_h_lp[kp] + s_h_l0[kpp]) +
				mu12kp*(-s_h_lm[km] + s_h_lp[km] - s_h_lmm[k0] + 3 * s_h_lm[k0] - 3 * s_h_lp[k0] + s_h_lpp[k0] - s_h_lmm[kp] + 3 * s_h_lm[kp] - 3 * s_h_lp[kp] + s_h_lpp[kp] - s_h_lm[kpp] + s_h_lp[kpp])
				) +
				(s_f1_l0[k0] + s_f1_lm[k0])*
			(
				mu22l0*(s_h_lm[km] - s_h_l0[km] + s_h_lmm[k0] - 5 * s_h_lm[k0] + 5 * s_h_l0[k0] - s_h_lp[k0] + s_h_lm[kp] - s_h_l0[kp]) +
				mu21l0*(s_h_lm[kmm] + s_h_l0[kmm] + s_h_lmm[km] - 3 * s_h_lm[km] - 3 * s_h_l0[km] + s_h_lp[km] - s_h_lmm[kp] + 3 * s_h_lm[kp] + 3 * s_h_l0[kp] - s_h_lp[kp] - s_h_lm[kpp] - s_h_l0[kpp])
				) +
				(s_f1_lp[k0] + s_f1_l0[k0])*
			(
				mu22lp*(-s_h_l0[km] + s_h_lp[km] - s_h_lm[k0] + 5 * s_h_l0[k0] - 5 * s_h_lp[k0] + s_h_lpp[k0] - s_h_l0[kp] + s_h_lp[kp]) +
				mu21lp*(-s_h_l0[kmm] - s_h_lp[kmm] - s_h_lm[km] + 3 * s_h_l0[km] + 3 * s_h_lp[km] - s_h_lpp[km] + s_h_lm[kp] - 3 * s_h_l0[kp] - 3 * s_h_lp[kp] + s_h_lpp[kp] + s_h_l0[kpp] + s_h_lp[kpp])
				);


		DATATYPE r_nonLinearRHS_divgrad =
			(s_f2_l0[k0] + s_f2_l0[km])*
			(
				mu11k0*(s_h_l0[km] - s_h_l0[k0]) +
				mu12k0*(s_h_lm[km] - s_h_lp[km] + s_h_lm[k0] - s_h_lp[k0])
				) +
				(s_f2_l0[kp] + s_f2_l0[k0])*
			(
				mu11kp*(-s_h_l0[k0] + s_h_l0[kp]) +
				mu12kp*(-s_h_lm[k0] + s_h_lp[k0] - s_h_lm[kp] + s_h_lp[kp])
				) +
				(s_f2_l0[k0] + s_f2_lm[k0])*
			(
				mu22l0*(s_h_lm[k0] - s_h_l0[k0]) +
				mu21l0*(s_h_lm[km] + s_h_l0[km] - s_h_lm[kp] - s_h_l0[kp])
				) +
				(s_f2_lp[k0] + s_f2_l0[k0])*
			(
				mu22lp*(-s_h_l0[k0] + s_h_lp[k0]) +
				mu21lp*(-s_h_l0[km] - s_h_lp[km] + s_h_l0[kp] + s_h_lp[kp])
				);

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


			// Note the terms in each multiplication, the l0,lp,lm, ... subscripts should match, except multiplying by s_df1_ and s_df2_ terms

			Jy.c = s_df1_l0[k0] * r_hyyy_lp_half*mu22lp + HYYY_LM_LP*r_f1_lp_half*mu22lp - s_df1_l0[k0] * r_hyyy_l0_half*mu22l0 - HYYY_L0_L0*r_f1_l0_half*mu22l0
				+ s_df2_l0[k0] * r_hy_lp_half*mu22lp + HY_LM_LP*r_f2_lp_half*mu22lp - s_df2_l0[k0] * r_hy_l0_half*mu22l0 - HY_L0_L0*r_f2_l0_half*mu22l0;

			Jy.b = -s_df1_lm[k0] * r_hyyy_l0_half*mu22l0 - HYYY_LM_L0*r_f1_l0_half*mu22l0 + HYYY_LMM_LP*r_f1_lp_half*mu22lp
				- s_df2_lm[k0] * r_hy_l0_half*mu22l0 - HY_LM_L0*r_f2_l0_half*mu22l0;


			Jy.d = s_df1_lp[k0] * r_hyyy_lp_half*mu22lp + HYYY_L0_LP*r_f1_lp_half*mu22lp - HYYY_LP_L0*r_f1_l0_half*mu22l0
				+ s_df2_lp[k0] * r_hy_lp_half*mu22lp + HY_L0_LP*r_f2_lp_half*mu22lp;

			Jy.e = r_f1_lp_half*HYYY_LP_LP*mu22lp;
			Jy.a = -r_f1_l0_half*HYYY_LMM_L0*mu22l0;

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

	template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE, boundary_condtion_type::IDs BC_Y0, boundary_condtion_type::IDs BC_YM> __device__
		void compute_Jy_and_F_row_subloop(
			reduced_device_workspace<DATATYPE> d_ws, indices_Jy &idx, dimensions dims,
			DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
			DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
			DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
			DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
			DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE],
			DATATYPE s_mu1_lm[LOAD_SIZE], DATATYPE s_mu1_l0[LOAD_SIZE], DATATYPE s_mu1_lp[LOAD_SIZE],
			DATATYPE s_mu2_lm[LOAD_SIZE], DATATYPE s_mu2_l0[LOAD_SIZE], DATATYPE s_mu2_lp[LOAD_SIZE])
	{
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
	}

	template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, int LOAD_SIZE, boundary_condtion_type::IDs BC_Y0, boundary_condtion_type::IDs BC_YM> __device__
		void compute_Jy_and_F_row_last_subloop(
			reduced_device_workspace<DATATYPE> d_ws, indices_Jy &idx, dimensions dims,
			DATATYPE s_h_lmm[LOAD_SIZE], DATATYPE s_h_lm[LOAD_SIZE], DATATYPE s_h_l0[LOAD_SIZE], DATATYPE s_h_lp[LOAD_SIZE], DATATYPE s_h_lpp[LOAD_SIZE],
			DATATYPE s_f1_lm[LOAD_SIZE], DATATYPE s_f1_l0[LOAD_SIZE], DATATYPE s_f1_lp[LOAD_SIZE],
			DATATYPE s_df1_lm[LOAD_SIZE], DATATYPE s_df1_l0[LOAD_SIZE], DATATYPE s_df1_lp[LOAD_SIZE],
			DATATYPE s_f2_lm[LOAD_SIZE], DATATYPE s_f2_l0[LOAD_SIZE], DATATYPE s_f2_lp[LOAD_SIZE],
			DATATYPE s_df2_lm[LOAD_SIZE], DATATYPE s_df2_l0[LOAD_SIZE], DATATYPE s_df2_lp[LOAD_SIZE],
			DATATYPE s_mu1_lm[LOAD_SIZE], DATATYPE s_mu1_l0[LOAD_SIZE], DATATYPE s_mu1_lp[LOAD_SIZE],
			DATATYPE s_mu2_lm[LOAD_SIZE], DATATYPE s_mu2_l0[LOAD_SIZE], DATATYPE s_mu2_lp[LOAD_SIZE], int sub_loop_length)
	{
		if (sub_loop_length > 0) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		if (sub_loop_length > 1) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		if (sub_loop_length > 2) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		if (sub_loop_length > 3) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		if (sub_loop_length > 4) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		if (sub_loop_length > 5) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		if (sub_loop_length > 6) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		if (sub_loop_length > 7) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		if (sub_loop_length > 8) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		if (sub_loop_length > 9) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		if (sub_loop_length > 10) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		if (sub_loop_length > 11) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		if (sub_loop_length > 12) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, s_mu1_lm, s_mu1_l0, s_mu1_lp, s_mu2_lm, s_mu2_l0, s_mu2_lp, idx, d_ws, dims);
		if (sub_loop_length > 13) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, s_mu1_l0, s_mu1_lp, s_mu1_lm, s_mu2_l0, s_mu2_lp, s_mu2_lm, idx, d_ws, dims);
		if (sub_loop_length > 14) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, s_mu1_lp, s_mu1_lm, s_mu1_l0, s_mu2_lp, s_mu2_lm, s_mu2_l0, idx, d_ws, dims);
		//if (sub_loop_length > 0) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		//if (sub_loop_length > 1) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		//if (sub_loop_length > 2) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		//if (sub_loop_length > 3) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		//if (sub_loop_length > 4) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		//if (sub_loop_length > 5) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		//if (sub_loop_length > 6) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		//if (sub_loop_length > 7) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		//if (sub_loop_length > 8) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		//if (sub_loop_length > 9) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		//if (sub_loop_length > 10) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		//if (sub_loop_length > 11) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lm, s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
		//if (sub_loop_length > 12) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_l0, s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_f1_lm, s_f1_l0, s_f1_lp, s_df1_lm, s_df1_l0, s_df1_lp, s_f2_lm, s_f2_l0, s_f2_lp, s_df2_lm, s_df2_l0, s_df2_lp, idx, d_ws, dims);
		//if (sub_loop_length > 13) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lp, s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_f1_l0, s_f1_lp, s_f1_lm, s_df1_l0, s_df1_lp, s_df1_lm, s_f2_l0, s_f2_lp, s_f2_lm, s_df2_l0, s_df2_lp, s_df2_lm, idx, d_ws, dims);
		//if (sub_loop_length > 14) compute_Jy_and_F_row_single<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(s_h_lpp, s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_f1_lp, s_f1_lm, s_f1_l0, s_df1_lp, s_df1_lm, s_df1_l0, s_f2_lp, s_f2_lm, s_f2_l0, s_df2_lp, s_df2_lm, s_df2_l0, idx, d_ws, dims);
	}

	template <typename DATATYPE, int BLOCK_SIZE, int THREAD_SIZE, bool FIRST_NEWTON_ITERATION, boundary_condtion_type::IDs BC_Y0, boundary_condtion_type::IDs BC_YM> __global__
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

		__shared__ DATATYPE s_mu1_lm[LOAD_SIZE];
		__shared__ DATATYPE s_mu1_l0[LOAD_SIZE];
		__shared__ DATATYPE s_mu1_lp[LOAD_SIZE];


		__shared__ DATATYPE s_mu2_lm[LOAD_SIZE];
		__shared__ DATATYPE s_mu2_l0[LOAD_SIZE];
		__shared__ DATATYPE s_mu2_lp[LOAD_SIZE];



		int x = blockIdx.x*THREAD_SIZE + threadIdx.x;
		int y = blockIdx.y*BLOCK_SIZE;

		indices_Jy idx;

		idx.shifted_padded_globalIdx_kpp = (y + 2 * PADDING2)*dims.n_pad + x;
		idx.globalIdx = (y)*dims.n + x;
		idx.y = y;

		//idx.valid_x_point = (x < dims.n);
		if (x < dims.n)
		{
			load_first_solution_row<DATATYPE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE>(s_h_lmm, s_h_lm, s_h_l0, s_h_lp,
				s_f1_lm, s_f1_l0,
				s_df1_lm, s_df1_l0,
				s_f2_lm, s_f2_l0,
				s_df2_lm, s_df2_l0,
				s_mu1_lm, s_mu1_l0,
				s_mu2_lm, s_mu2_l0,
				idx, d_ws, dims);

			if (blockIdx.y < gridDim.y - 1)
				compute_Jy_and_F_row_subloop<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(
					d_ws, idx, dims,
					s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp,
					s_f1_lm, s_f1_l0, s_f1_lp,
					s_df1_lm, s_df1_l0, s_df1_lp,
					s_f2_lm, s_f2_l0, s_f2_lp,
					s_df2_lm, s_df2_l0, s_df2_lp,
					s_mu1_lm, s_mu1_l0, s_mu1_lp,
					s_mu2_lm, s_mu2_l0, s_mu2_lp);
			else
				compute_Jy_and_F_row_last_subloop<DATATYPE, BLOCK_SIZE, THREAD_SIZE, FIRST_NEWTON_ITERATION, LOAD_SIZE, BC_Y0, BC_YM>(
					d_ws, idx, dims,
					s_h_lmm, s_h_lm, s_h_l0, s_h_lp, s_h_lpp,
					s_f1_lm, s_f1_l0, s_f1_lp,
					s_df1_lm, s_df1_l0, s_df1_lp,
					s_f2_lm, s_f2_l0, s_f2_lp,
					s_df2_lm, s_df2_l0, s_df2_lp,
					s_mu1_lm, s_mu1_l0, s_mu1_lp,
					s_mu2_lm, s_mu2_l0, s_mu2_lp,
					dims.Jy_F_last_y_block_size);

		}
	}
}
#endif