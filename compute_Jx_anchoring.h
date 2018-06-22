#ifndef LST_TYPE_2_JX
#define LST_TYPE_2_JX

namespace compute_Jx_anchoring
{

	int const PADDING_SIZE = 2;

#include "cuda_runtime.h"
	template <typename DATATYPE, int BLOCK_SIZE, bool FIRST_NEWTON_ITERATION> __device__
		void load_Jx_block
		(reduced_device_workspace<DATATYPE> d_ws, DATATYPE  s_h[BLOCK_SIZE + 4][BLOCK_SIZE + 1],
			DATATYPE  s_f1[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1], DATATYPE  s_df1[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1],
			DATATYPE  s_f2[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1], DATATYPE  s_df2[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1],
			DATATYPE  s_mu1[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1],
			int const globalIdx_padded_shifted, int const simple_last_block_x_size)
	{

		DATATYPE *d_h;

		// use old time step solution on FIRST_NEWTON_ITERATION
		(FIRST_NEWTON_ITERATION) ? (d_h = d_ws.h) : (d_h = d_ws.h_guess);

		int load_size;

		(blockIdx.x < gridDim.x - 1) ? (load_size = BLOCK_SIZE) : (load_size = simple_last_block_x_size);

		// Two (2) cases for loading (thread_size + 2*PADDING_SIZE) cells x points 
		//		Case 1: load_size >= 2*PADDING_SIZE - each data row loaded first using 'load_size' threads then '2*PADDING_SIZE' threads
		//		Case 2: load_size <  2*PADDING_SIZE - each data row loaded using 'load_size' threads, '(load_size + 2 * PADDING_SIZE)/load_size + 1' times

		if (load_size >= 2 * PADDING_SIZE)
		{

			s_h[threadIdx.x][threadIdx.y] = d_h[globalIdx_padded_shifted];
			s_f1[threadIdx.x][threadIdx.y] = d_ws.f1[globalIdx_padded_shifted];
			s_df1[threadIdx.x][threadIdx.y] = d_ws.df1[globalIdx_padded_shifted];

			s_f2[threadIdx.x][threadIdx.y] = d_ws.f2[globalIdx_padded_shifted];
			s_df2[threadIdx.x][threadIdx.y] = d_ws.df2[globalIdx_padded_shifted];

			s_mu1[threadIdx.x][threadIdx.y] = d_ws.mu1[globalIdx_padded_shifted];

			if (threadIdx.x < 2 * PADDING_SIZE) {

				s_h[threadIdx.x + load_size][threadIdx.y] = d_h[globalIdx_padded_shifted + load_size];

				s_f1[threadIdx.x + load_size][threadIdx.y] = d_ws.f1[globalIdx_padded_shifted + load_size];
				s_df1[threadIdx.x + load_size][threadIdx.y] = d_ws.df1[globalIdx_padded_shifted + load_size];

				s_f2[threadIdx.x + load_size][threadIdx.y] = d_ws.f2[globalIdx_padded_shifted + load_size];
				s_df2[threadIdx.x + load_size][threadIdx.y] = d_ws.df2[globalIdx_padded_shifted + load_size];

				s_mu1[threadIdx.x + load_size][threadIdx.y] = d_ws.mu1[globalIdx_padded_shifted + load_size];

			}

		}
		else
		{
			for (int i = 0; i < load_size + 2 * PADDING_SIZE; i += load_size)
			{

				s_h[threadIdx.x + i][threadIdx.y] = d_h[globalIdx_padded_shifted + i];

				s_f1[threadIdx.x + i][threadIdx.y] = d_ws.f1[globalIdx_padded_shifted + i];
				s_df1[threadIdx.x + i][threadIdx.y] = d_ws.df1[globalIdx_padded_shifted + i];

				s_f2[threadIdx.x + i][threadIdx.y] = d_ws.f2[globalIdx_padded_shifted + i];
				s_df2[threadIdx.x + i][threadIdx.y] = d_ws.df2[globalIdx_padded_shifted + i];


				s_mu1[threadIdx.x + i][threadIdx.y] = d_ws.mu1[globalIdx_padded_shifted + i];
			}
		}
	}
	template <typename DATATYPE, int BLOCK_SIZE, bool FIRST_NEWTON_ITERATION,
		boundary_condtion_type::IDs BC_X0, boundary_condtion_type::IDs BC_XN> __global__
		void compute_Jx(reduced_device_workspace<DATATYPE> d_ws, dimensions dims)
	{


		__shared__ DATATYPE  s_h[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1];
		__shared__ DATATYPE s_f1[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1];
		__shared__ DATATYPE s_f2[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1];
		__shared__ DATATYPE s_df1[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1];
		__shared__ DATATYPE s_df2[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1];

		__shared__ DATATYPE s_mu1[BLOCK_SIZE + 2 * PADDING_SIZE][BLOCK_SIZE + 1];

		int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

		DATATYPE *d_h;

		if (x < dims.n && y < dims.m)
		{


			// index of padded solution shifted by negative PADDING_SIZE = 2
			int globalIdx_padded_shifted = (y + PADDING_SIZE)*dims.n_pad + x;

			load_Jx_block<DATATYPE, BLOCK_SIZE, FIRST_NEWTON_ITERATION>(d_ws, s_h, s_f1, s_df1, s_f2, s_df2, s_mu1, globalIdx_padded_shifted, dims.simple_last_block_x_size);

			__syncthreads();

			penta_diag_row<DATATYPE> Jx;
			bool custom_Jx_row;

			DATATYPE  HXXX_KP_K0;		DATATYPE  HXXX_KP_KP;
			DATATYPE  HXXX_K0_K0;		DATATYPE  HXXX_K0_KP;
			DATATYPE  HXXX_KM_K0;		DATATYPE  HXXX_KM_KP;
			DATATYPE  HXXX_KMM_K0;      DATATYPE  HXXX_KMM_KP;

			DATATYPE  HX_K0_K0;			DATATYPE  HX_K0_KP;
			DATATYPE  HX_KM_K0;			DATATYPE  HX_KM_KP;

			if (2 < x && x < dims.n - 3) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition      ::INTERIOR, BC_X0, BC_XN>(HXXX_KP_K0, HXXX_KP_KP, HXXX_K0_K0, HXXX_K0_KP, HXXX_KM_K0, HXXX_KM_KP, HXXX_KMM_K0, HXXX_KMM_KP, HX_K0_K0, HX_K0_KP, HX_KM_K0, HX_KM_KP, Jx, custom_Jx_row);

			if (x == 0) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::FIRST, BC_X0, BC_XN>(HXXX_KP_K0, HXXX_KP_KP, HXXX_K0_K0, HXXX_K0_KP, HXXX_KM_K0, HXXX_KM_KP, HXXX_KMM_K0, HXXX_KMM_KP, HX_K0_K0, HX_K0_KP, HX_KM_K0, HX_KM_KP, Jx, custom_Jx_row);
			if (x == 1) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::SECOND, BC_X0, BC_XN>(HXXX_KP_K0, HXXX_KP_KP, HXXX_K0_K0, HXXX_K0_KP, HXXX_KM_K0, HXXX_KM_KP, HXXX_KMM_K0, HXXX_KMM_KP, HX_K0_K0, HX_K0_KP, HX_KM_K0, HX_KM_KP, Jx, custom_Jx_row);
			if (x == 2) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::THIRD, BC_X0, BC_XN>(HXXX_KP_K0, HXXX_KP_KP, HXXX_K0_K0, HXXX_K0_KP, HXXX_KM_K0, HXXX_KM_KP, HXXX_KMM_K0, HXXX_KMM_KP, HX_K0_K0, HX_K0_KP, HX_KM_K0, HX_KM_KP, Jx, custom_Jx_row);
			if (x == dims.n - 1) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::FIRST_LAST, BC_X0, BC_XN>(HXXX_KP_K0, HXXX_KP_KP, HXXX_K0_K0, HXXX_K0_KP, HXXX_KM_K0, HXXX_KM_KP, HXXX_KMM_K0, HXXX_KMM_KP, HX_K0_K0, HX_K0_KP, HX_KM_K0, HX_KM_KP, Jx, custom_Jx_row);
			if (x == dims.n - 2) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::SECOND_LAST, BC_X0, BC_XN>(HXXX_KP_K0, HXXX_KP_KP, HXXX_K0_K0, HXXX_K0_KP, HXXX_KM_K0, HXXX_KM_KP, HXXX_KMM_K0, HXXX_KMM_KP, HX_K0_K0, HX_K0_KP, HX_KM_K0, HX_KM_KP, Jx, custom_Jx_row);
			if (x == dims.n - 3) boundary_condition::lhs_matrix_J<DATATYPE, boundary_condtion_postition::THIRD_LAST, BC_X0, BC_XN>(HXXX_KP_K0, HXXX_KP_KP, HXXX_K0_K0, HXXX_K0_KP, HXXX_KM_K0, HXXX_KM_KP, HXXX_KMM_K0, HXXX_KMM_KP, HX_K0_K0, HX_K0_KP, HX_KM_K0, HX_KM_KP, Jx, custom_Jx_row);


			int globalIdx_transpose = x*dims.m + y;
			if (!custom_Jx_row)
			{
				int k0 = threadIdx.x + 2;
				int kp = k0 + 1;
				int kpp = k0 + 2;
				int km = k0 - 1;
				int kmm = k0 - 2;

				DATATYPE r_f1_k0_half = s_f1[k0][threadIdx.y] + s_f1[km][threadIdx.y];
				DATATYPE r_f1_kp_half = s_f1[kp][threadIdx.y] + s_f1[k0][threadIdx.y];

				DATATYPE r_f2_k0_half = s_f2[k0][threadIdx.y] + s_f2[km][threadIdx.y];
				DATATYPE r_f2_kp_half = s_f2[kp][threadIdx.y] + s_f2[k0][threadIdx.y];

				DATATYPE r_hx_kp_half = s_h[kp][threadIdx.y] - s_h[k0][threadIdx.y];
				DATATYPE r_hx_k0_half = s_h[k0][threadIdx.y] - s_h[km][threadIdx.y];

				DATATYPE r_hxxx_kp_half = s_h[kpp][threadIdx.y] - 3.0*r_hx_kp_half - s_h[km][threadIdx.y];
				DATATYPE r_hxxx_k0_half = s_h[kp][threadIdx.y] - 3.0*r_hx_k0_half - s_h[kmm][threadIdx.y];



				//DATATYPE const lambda = 1.8;
				//DATATYPE const nu = 1.3;

				DATATYPE const lambda = 0.75;
				DATATYPE const nu = 1.0- lambda;
				DATATYPE mu11kp = lambda + nu*(s_mu1[kp][threadIdx.y] + s_mu1[k0][threadIdx.y]);
				DATATYPE mu11k0 = lambda + nu*(s_mu1[k0][threadIdx.y] + s_mu1[km][threadIdx.y]);



				Jx.c = s_df1[k0][threadIdx.y] * r_hxxx_kp_half*mu11kp + HXXX_KM_KP*r_f1_kp_half*mu11kp - s_df1[k0][threadIdx.y] * mu11k0* r_hxxx_k0_half - HXXX_K0_K0*r_f1_k0_half*mu11k0
					+ s_df2[k0][threadIdx.y] * r_hx_kp_half*mu11kp + HX_KM_KP*r_f2_kp_half*mu11kp - s_df2[k0][threadIdx.y] * mu11k0*   r_hx_k0_half - HX_K0_K0*r_f2_k0_half*mu11k0;

				Jx.b = -s_df1[km][threadIdx.y] * r_hxxx_k0_half*mu11k0 - HXXX_KM_K0*r_f1_k0_half*mu11k0 + HXXX_KMM_KP*r_f1_kp_half*mu11kp
					- s_df2[km][threadIdx.y] * r_hx_k0_half*mu11k0 - HX_KM_K0*r_f2_k0_half*mu11k0;


				Jx.d = s_df1[kp][threadIdx.y] * r_hxxx_kp_half*mu11kp + HXXX_K0_KP*r_f1_kp_half*mu11kp - HXXX_KP_K0*r_f1_k0_half*mu11k0
					+ s_df2[kp][threadIdx.y] * r_hx_kp_half*mu11kp + HX_K0_KP*r_f2_kp_half*mu11kp;

				Jx.e = r_f1_kp_half*HXXX_KP_KP*mu11kp;
				Jx.a = -r_f1_k0_half*HXXX_KMM_K0*mu11k0;
			}
			else
				d_ws.w_transpose[globalIdx_transpose] = Jx.f;


			d_ws.Jx_a[globalIdx_transpose] = Jx.a;
			d_ws.Jx_b[globalIdx_transpose] = Jx.b;
			d_ws.Jx_c[globalIdx_transpose] = 1.0 + Jx.c;
			d_ws.Jx_d[globalIdx_transpose] = Jx.d;
			d_ws.Jx_e[globalIdx_transpose] = Jx.e;



		}



	}

}

#endif