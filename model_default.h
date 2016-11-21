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

#ifndef MODEL_DEFAULT
#define MODEL_DEFAULT

// Imports base model_parameters template.
#include "model_template.h"

namespace model_default
{

	const model_id ID = model_id::DEFAULT;

	template <typename DATATYPE > struct model_parameters<DATATYPE , ID>
	{

		// Main parameters
		DATATYPE cK;
		DATATYPE cC;
		DATATYPE cN;
		DATATYPE beta;
		DATATYPE b;
		DATATYPE w;
		DATATYPE ww;

		// Derived parameters common to each spacial point,
		// precomputed on CPU to remove redundant GPU computations.
		// Only required if expression for nonlinear functions 
		// are complicated and contain many fixed terms.
		DATATYPE inv_w;
		DATATYPE beta2;
		DATATYPE w2;
		DATATYPE two_b;
		DATATYPE two_w;
		DATATYPE three_b;
		DATATYPE scaled_cN;
		DATATYPE cK_b2;
		DATATYPE scaled_cK_b3;

		void compute_derived_parameters()
		{
			inv_w = 1 / w;;
			beta2 = beta*beta;
			w2 = w*w;
			two_b = 2 * b;
			three_b = 3 * b;
			two_w = 2 * w;
			scaled_cN = 0.25*cN*inv_w;
			cK_b2 = cK*b*b;
			scaled_cK_b3 = 3.0*cK*b*b*b;
		}

	};

		
	// Dummy subroutine template created so switch in subroutine
	// 'newton_iterative_method::compute_nonlinear_functions()'
	// has valid compile time function calls for switch paths 
    // that are not current model ID. 
	// Note: By construction, subroutine is never called at runtime. 
	template <typename DATATYPE, model_id MODEL_ID> __device__ __forceinline__
		void nonlinear_functions
		(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h , model_parameters<DATATYPE, MODEL_ID> modelParas, int h_index);


	// Definition of nonlinear functions for this model ID.
	template <typename DATATYPE, model_id MODEL_ID=ID> __device__ __forceinline__
		void nonlinear_functions(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h , model_parameters<DATATYPE, ID > modelParas, int h_index)
	{

		DATATYPE cC = modelParas.cC;
		DATATYPE cN = modelParas.cN;
		DATATYPE cK = modelParas.cK;
		DATATYPE beta = modelParas.beta;
		DATATYPE b = modelParas.b;
		DATATYPE w = modelParas.w;

		DATATYPE invW = modelParas.inv_w;
		DATATYPE beta2 = modelParas.beta2;
		DATATYPE w2 = modelParas.w2;
		DATATYPE two_b = modelParas.two_b;
		DATATYPE two_w = modelParas.two_w;
		DATATYPE three_b = modelParas.three_b;
		DATATYPE scaled_cN = modelParas.scaled_cN;
		DATATYPE cK_b2 = modelParas.cK_b2;
		DATATYPE scaled_cK_b3 = modelParas.scaled_cK_b3;


		DATATYPE h = d_h[h_index];
		DATATYPE h2 = h*h;
		DATATYPE h3 = h2*h;

		d_ws.f1[h_index] = cC*h3;

		d_ws.df1[h_index] = 3.0*cC*h2;

		DATATYPE tau = tanh((h - 2 * b)*invW);
		DATATYPE kappa = tau - 1;;

		DATATYPE eta = h2 + beta*beta;
		DATATYPE eta2 = eta*eta;
		DATATYPE invEta = 1.0 / eta;

		DATATYPE shareNematic = scaled_cN*(tau+1)*(tau+1)*invEta*invEta*invEta*h3;
		DATATYPE invH = 1.0/h;

		d_ws.f2[h_index] = -h*shareNematic*(h2*two_w - w*eta + h*eta*(tau-1));
		d_ws.f2[h_index] += cK_b2*(2-three_b*invH);
			
		d_ws.df2[h_index] = invW*invEta*shareNematic*( 4*w2*(3*h2*h2 - 4*h2*eta + eta2) + h*eta*( 8*h2*w + 2*h*eta - 7*w*eta)*kappa+3*h2*eta2*kappa*kappa);
		d_ws.df2[h_index] += scaled_cK_b3*invH*invH;

	}


}

#endif