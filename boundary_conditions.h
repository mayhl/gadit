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

#ifndef BOUNDARY_CONDITION
#define BOUNDARY_CONDITION

#include "dimensions.h"
#include "work_space.h"
#include "parameters.h"


namespace boundary_condition
{

	// Name: Boundary Conditions
	// Version: 1.0 
	// 
	// Input:	d_h - a reference to array d_ws.h or d_ws.h_guess, depending on  
	//
	//			d_ws - struct containing array references to nonlinear functions (f1 and f2) and
	//                 their derivatives (df1 and df2). NOTE: Don not use references to h or h_guess.
	//
	//			dims  - struct containing dimension of problem (n,m) and padded dimensions (n_pad,m_pad)
	//
	//
	// Description: Routines to apply boundary condition to MxN LHS matrices of penta-diagonal matrices, Jx and Jy;
	//				and RHS matrix, F. Boundary conditions are implemented in two ways:
	//					1) The solution, h; the non-linear function values, f1 and f2; and its derivative w.r.t. h,
	//					   df1 and df2; are padded by two (2) elements i.e. a (M+2)x(Nx2) matrix. The values of the padded
	//					   elements are set to satisfy the boundary conditions.
	//
	//					   NOTE: a) The stencil used to calculate derivatives  

#pragma region rhs_vector_bc

	// Implementation info
	// 
	//    BC			   Info					Implementation
	// --------			---------				--------------
	// SYMMETRIC		h_s = h_sss= 0		h(-ds/2)=h(ds/2) and h(-3/2*ds)=h(3/2*ds) 


	// Modifying X0 boundary ghost points to satisfy boundary conditions
	template <typename DATATYPE, boundary_condtion_type BC_X > __device__ __forceinline__
		void rhs_vector_X0_row(reduced_device_workspace<DATATYPE> d_ws, dimensions dims,
		DATATYPE* d_h, int  const  k0, int  const  l0, int const k0_l0 )
	{

			switch (BC_X)
			{
			case boundary_condtion_type::SYMMETRIC:
				if (k0 == 0)
				{

					int km_l0 = k0_l0 - 1;
					d_h[km_l0] = d_h[k0_l0];

					d_ws.f1[km_l0] = d_ws.f1[k0_l0];
					d_ws.df1[km_l0] = d_ws.df1[k0_l0];


					d_ws.f2[km_l0] = d_ws.f2[k0_l0];
					d_ws.df2[km_l0] = d_ws.df2[k0_l0];
				}


				if (k0 == 1)
				{
					int kmmm_l0 = k0_l0 - 3;
					d_h[kmmm_l0] = d_h[k0_l0];
				}

		

				break;


			}

		}

	// Modifying XN boundary ghost points to satisfy boundary conditions
	template <typename DATATYPE, boundary_condtion_type BC_X > __device__ __forceinline__
		void rhs_vector_XN_row(reduced_device_workspace<DATATYPE> d_ws, dimensions dims,
		DATATYPE* d_h, int  const  k0, int  const  l0, int const k0_l0)
	{

			switch (BC_X)
			{
			case boundary_condtion_type::SYMMETRIC:
		
				if (k0 == (dims.n-1))
				{

					int kp_l0 = k0_l0 + 1;
					d_h[kp_l0] = d_h[k0_l0];

					d_ws.f1[kp_l0] = d_ws.f1[k0_l0];
					d_ws.df1[kp_l0] = d_ws.df1[k0_l0];


					d_ws.f2[kp_l0] = d_ws.f2[k0_l0];
					d_ws.df2[kp_l0] = d_ws.df2[k0_l0];
				}


				if (k0 == (dims.n - 2))
				{
					int kppp_l0 = k0_l0 + 3;
					d_h[kppp_l0] = d_h[k0_l0];
				}

				
				break;

			}
		}

	// Modifying Y0 boundary ghost points to satisfy boundary conditions
	template <typename DATATYPE, boundary_condtion_type BC_Y0 > __device__ __forceinline__
		void rhs_vector_Y0_column(reduced_device_workspace<DATATYPE> d_ws, dimensions dims,
		DATATYPE* d_h, int  const  k, int  const  l0, int const k0_l0)
	{

			switch (BC_Y0)
			{
			case boundary_condtion_type::SYMMETRIC:
				if (l0 == 0)
				{

					int k0_lm = k0_l0 - dims.n_pad;
					d_h[k0_lm] = d_h[k0_l0];

					d_ws.f1[k0_lm] = d_ws.f1[k0_l0];
					d_ws.df1[k0_lm] = d_ws.df1[k0_l0];


					d_ws.f2[k0_lm] = d_ws.f2[k0_l0];
					d_ws.df2[k0_lm] = d_ws.df2[k0_l0];
				}


				if (l0 == 1)
				{
					int k0_lmmm = k0_l0 - 3 * dims.n_pad;
					d_h[k0_lmmm] = d_h[k0_l0];
				}
				break;

			}
	}

	// Modifying YM boundary ghost points to satisfy boundary conditions
	template <typename DATATYPE, boundary_condtion_type BC_Y > __device__ __forceinline__
		void rhs_vector_YM_column(reduced_device_workspace<DATATYPE> d_ws, dimensions dims,
		DATATYPE* d_h, int  const  k0, int  const  l0, int const k0_l0)
	{

			switch (BC_Y)
			{
			case boundary_condtion_type::SYMMETRIC:
					if (l0 == dims.m - 1)
					{
						int k0_lp = k0_l0 + dims.n_pad;
						d_h[k0_lp] = d_h[k0_l0];

						d_ws.f1[k0_lp] = d_ws.f1[k0_l0];
						d_ws.df1[k0_lp] = d_ws.df1[k0_l0];


						d_ws.f2[k0_lp] = d_ws.f2[k0_l0];
						d_ws.df2[k0_lp] = d_ws.df2[k0_l0];

					}

					if (l0 == dims.m - 2)
					{
						int k0_lppp = k0_l0 + 3 * dims.n_pad;
						d_h[k0_lppp] = d_h[k0_l0];
					}
				break;
			}


		}

	
	// Modifying corner ghost points to satisfy boundary conditions. 
	template <typename DATATYPE, boundary_condtion_type BC_X0, boundary_condtion_type BC_XN, boundary_condtion_type BC_Y0, boundary_condtion_type BC_YM> __device__ __forceinline__
		void rhs_vector_corner_blocks
		(reduced_device_workspace<DATATYPE> d_ws, dimensions dims, DATATYPE* d_h, int const  k0, int const l0, int const k0_l0)
	{

			int km_lm = k0_l0 - dims.n_pad - 1;
			switch (BC_X0)
			{
			case boundary_condtion_type::SYMMETRIC:
				switch (BC_Y0)
				{
				case boundary_condtion_type::SYMMETRIC:
					if ((k0 == 0) && (l0 == 0)) d_h[km_lm] = d_h[k0_l0];
					break;
				}

				break;
			}

			int km_lp = k0_l0 + dims.n_pad - 1;
			switch (BC_X0)
			{
			case boundary_condtion_type::SYMMETRIC:
				switch (BC_YM)
				{
				case boundary_condtion_type::SYMMETRIC:
					if ((k0 == 0) && (l0 == dims.m - 1))  d_h[km_lp] = d_h[k0_l0];
					break;
				}

				break;
			}


			int kp_lm = k0_l0 - dims.n_pad + 1;
			switch (BC_XN)
			{
			case boundary_condtion_type::SYMMETRIC:
				switch (BC_Y0)
				{
				case boundary_condtion_type::SYMMETRIC:
					if ((k0 == dims.n - 1) && (l0 == 0))  d_h[kp_lm] = d_h[k0_l0];

					break;
				}

				break;
			}

			int kp_lp = k0_l0 + dims.n_pad + 1;
			switch (BC_XN)
			{
			case boundary_condtion_type::SYMMETRIC:
				switch (BC_YM)
				{
				case boundary_condtion_type::SYMMETRIC:
					if ((k0 == dims.n - 1) && (l0 == dims.m - 1))  d_h[kp_lp] = d_h[k0_l0];

					break;
				}

				break;
			}
		}

#pragma endregion rhs_vector_bc

	// modify stencils for matrix J at boundaries.
	template <typename DATATYPE, bc_postition BC_POS ,boundary_condtion_type BC_START, boundary_condtion_type BC_END> __device__ __forceinline__
		void lhs_matrix_J
			(DATATYPE  &HSSS_LP_L0,		DATATYPE  &HSSS_LP_LP,
			DATATYPE  &HSSS_L0_L0,		DATATYPE  &HSSS_L0_LP,
			DATATYPE  &HSSS_LM_L0,		DATATYPE  &HSSS_LM_LP,
			DATATYPE  &HSSS_LMM_L0,		DATATYPE  &HSSS_LMM_LP,
			DATATYPE  &HS_L0_L0,		DATATYPE  &HS_L0_LP,
			DATATYPE  &HS_LM_L0,		DATATYPE  &HS_LM_LP,
			penta_diag_row<DATATYPE> &J , bool &custom_J_row )
			{

		custom_J_row = false;

		if (BC_POS == bc_postition::INTERIOR)
		{
			HSSS_LP_L0 = 1.0;		HSSS_LP_LP = 1.0;
			HSSS_L0_L0 = -3.0;		HSSS_L0_LP = -3.0;
			HSSS_LM_L0 = 3.0;		HSSS_LM_LP = 3.0;
			HSSS_LMM_L0 = -1.0;		HSSS_LMM_LP = -1.0;

			HS_L0_L0 = 1.0;			HS_L0_LP = 1.0;
			HS_LM_L0 = -1.0;		HS_LM_LP = -1.0;
		}
		else
		{
	

			if ((BC_POS == bc_postition::FIRST) || (BC_POS == bc_postition::SECOND) || (BC_POS == bc_postition::THIRD))
			{
				switch (BC_START)
				{
					case boundary_condtion_type::SYMMETRIC:
						switch (BC_POS)
						{

						case bc_postition::FIRST:
							HSSS_LP_L0 = 0;			HSSS_LP_LP = 1.0;
							HSSS_L0_L0 = 0;			HSSS_L0_LP = -3.0;
							HSSS_LM_L0 = 0;			HSSS_LM_LP = 2.0;
							HSSS_LMM_L0 = 0;		HSSS_LMM_LP = 0;

							HS_L0_L0 = 0;			HS_L0_LP = 1.0;
							HS_LM_L0 = 0;			HS_LM_LP = -1.0;
							break;
						case bc_postition::SECOND:
							HSSS_LP_L0 = 1.0;		HSSS_LP_LP = 1.0;
							HSSS_L0_L0 = -3.0;		HSSS_L0_LP = -3.0;
							HSSS_LM_L0 = 2.0;		HSSS_LM_LP = 3.0;
							HSSS_LMM_L0 = 0;		HSSS_LMM_LP = -1.0;

							HS_L0_L0 = 1.0;			HS_L0_LP = 1.0;
							HS_LM_L0 = -1.0;		HS_LM_LP = -1.0;
							break;
						case bc_postition::THIRD:
							HSSS_LP_L0 = 1.0;		HSSS_LP_LP = 1.0;
							HSSS_L0_L0 = -3.0;		HSSS_L0_LP = -3.0;
							HSSS_LM_L0 = 3.0;		HSSS_LM_LP = 3.0;
							HSSS_LMM_L0 = -1.0;		HSSS_LMM_LP = -1.0;

							HS_L0_L0 = 1.0;			HS_L0_LP = 1.0;
							HS_LM_L0 = -1.0;		HS_LM_LP = -1.0;
							break;
						}
						break;
				}
			}
			else if ((BC_POS == bc_postition::FIRST_LAST) || (BC_POS == bc_postition::SECOND_LAST) || (BC_POS == bc_postition::THIRD_LAST))
			{
				switch (BC_END)
				{
				case boundary_condtion_type::SYMMETRIC:
					switch (BC_POS)
					{
					case bc_postition::FIRST_LAST:
						HSSS_LP_L0 = 0.0;		HSSS_LP_LP = 0;
						HSSS_L0_L0 = -2.0;		HSSS_L0_LP = 0;
						HSSS_LM_L0 = 3.0;		HSSS_LM_LP = 0;
						HSSS_LMM_L0 = -1.0;		HSSS_LMM_LP = 0;

						HS_L0_L0 = 1.0;			HS_L0_LP = 0.0;
						HS_LM_L0 = -1.0;		HS_LM_LP = 0.0;
						break;
					case bc_postition::SECOND_LAST:
						HSSS_LP_L0 = 1.0;		HSSS_LP_LP = 0.0;
						HSSS_L0_L0 = -3.0;		HSSS_L0_LP = -2.0;
						HSSS_LM_L0 = 3.0;		HSSS_LM_LP = 3.0;
						HSSS_LMM_L0 = -1.0;		HSSS_LMM_LP = -1.0;

						HS_L0_L0 = 1.0;			HS_L0_LP = 1.0;
						HS_LM_L0 = -1.0;		HS_LM_LP = -1.0;
						break;
					case bc_postition::THIRD_LAST:
						HSSS_LP_L0 = 1.0;		HSSS_LP_LP = 1.0;
						HSSS_L0_L0 = -3.0;		HSSS_L0_LP = -3.0;
						HSSS_LM_L0 = 3.0;		HSSS_LM_LP = 3.0;
						HSSS_LMM_L0 = -1.0;		HSSS_LMM_LP = -1.0;

						HS_L0_L0 = 1.0;			HS_L0_LP = 1.0;
						HS_LM_L0 = -1.0;		HS_LM_LP = -1.0;
						break;
					}
					break;

				}
			}
				
		}

	}
}

#endif