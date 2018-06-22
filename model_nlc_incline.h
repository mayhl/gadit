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

#ifndef MODEL_NLC_INCLINE
#define MODEL_NLC_INCLINE

// Imports base model_parameters template.
#include "solver_template.h"

namespace model_nlc_incline
{
	// If copying to make new model, change to new model id.
	const model::id ID = model::NLC_INCLINE;

	template <typename DATATYPE > struct model_parameters<DATATYPE, ID>
	{

		// Main parameters
		DATATYPE cD;
		DATATYPE cC;
		DATATYPE cN;
		DATATYPE beta;
		DATATYPE b;
		DATATYPE w;

		// Derived parameters common to each spacial point,
		// precomputed on CPU to remove redundant GPU computations.
		// Only required if expression for nonlinear functions 
		// are complicated and contain many fixed terms.
		DATATYPE inv_w;
		DATATYPE beta2;
		DATATYPE w2;
		DATATYPE two_w;
		DATATYPE three_b;
		DATATYPE scaled_cN;

		std::string to_string()
		{
			std::string output;
			output = format_parameter_output::make_title("NLC Model");

			output += "cD   = " + format_parameter_output::datatype(this->cD) + "\n";
			output += "cC   = " + format_parameter_output::datatype(this->cC) + "\n";
			output += "cN   = " + format_parameter_output::datatype(this->cN) + "\n";
			output += "beta = " + format_parameter_output::datatype(this->beta) + "\n";
			output += "b    = " + format_parameter_output::datatype(this->b) + "\n";
			output += "w    = " + format_parameter_output::datatype(this->w) + "\n";
			output += "\n";

			return output;
		}

		void compute_derived_parameters()
		{
			inv_w = 1 / w;;
			beta2 = beta*beta;
			w2 = w*w;
			three_b = 3 * b;
			two_w = 2 * w;
			scaled_cN = 0.25*cN*inv_w;
		}

		DATATYPE getGrowthRate(DATATYPE h, DATATYPE q)
		{
			DATATYPE gRate;
			DATATYPE f1;
			DATATYPE f2;

			DATATYPE h2 = h*h;
			DATATYPE h3 = h2*h;

			DATATYPE tau = tanh((h - 2 * b)*inv_w);
			DATATYPE kappa = tau - 1;;

			DATATYPE eta = h2 + beta*beta;
			DATATYPE eta2 = eta*eta;
			DATATYPE invEta = 1.0 / eta;

			DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
			DATATYPE invH = 1.0 / h;

			f2 = -h*shareNematic*(h2*two_w - w*eta + h*eta*(tau - 1));
			f2 += -cD*h3;

			f1 = cC*h3;

			DATATYPE q2 = q*q;
			gRate = -(f1*q2 - f2)*q2;

			return gRate;


		}
		DATATYPE getMaxGrowthMode(DATATYPE h)
		{
			DATATYPE qm;
			DATATYPE f1;
			DATATYPE f2;

			DATATYPE h2 = h*h;
			DATATYPE h3 = h2*h;

			DATATYPE tau = tanh((h - 2 * b)*inv_w);
			DATATYPE kappa = tau - 1;;

			DATATYPE eta = h2 + beta*beta;
			DATATYPE invEta = 1.0 / eta;

			DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
			DATATYPE invH = 1.0 / h;

			f2 = -h*shareNematic*(h2*two_w - w*eta + h*eta*(tau - 1));
			f2 += -cD*h3;

			f1 = cC*h3;

			qm = sqrt(f2 / (2.0*f1));

			return qm;
		}

	};


	// Dummy subroutine template created so switch in subroutine
	// 'newton_iterative_method::compute_nonlinear_functions()'
	// has valid compile time function calls for switch paths 
	// that are not current model ID. 
	// Note: By construction, subroutine is never called at runtime. 
	template <typename DATATYPE, model::id MODEL_ID> __device__ __forceinline__
		void nonlinear_functions
		(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h, model_parameters<DATATYPE, MODEL_ID> modelParas, int h_index);


	// Definition of nonlinear functions for this model ID.
	template <typename DATATYPE, model::id MODEL_ID = ID> __device__ __forceinline__
		void nonlinear_functions(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h, model_parameters<DATATYPE, ID > modelParas, int h_index)
	{

		DATATYPE cC = modelParas.cC;
		DATATYPE cD = modelParas.cD;
		DATATYPE cN = modelParas.cN;
		DATATYPE beta = modelParas.beta;
		DATATYPE b = modelParas.b;
		DATATYPE w = modelParas.w;

		DATATYPE invW = modelParas.inv_w;
		DATATYPE beta2 = modelParas.beta2;
		DATATYPE w2 = modelParas.w2;
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

		DATATYPE eta = h2 + beta2;
		DATATYPE eta2 = eta*eta;
		DATATYPE invEta = 1.0 / eta;

		DATATYPE shareNematic = scaled_cN*(tau + 1)*(tau + 1)*invEta*invEta*invEta*h3;
		DATATYPE invH = 1.0 / h;

		d_ws.f2[h_index] = -h*shareNematic*(h2*two_w - w*eta + h*eta*kappa);
		d_ws.f2[h_index] += cK_b2*(2 - three_b*invH);

		d_ws.df2[h_index] = invW*invEta*shareNematic*(4 * w2*(3 * h2*h2 - 4 * h2*eta + eta2) + h*eta*(8 * h2*w + 2 * h*eta - 7 * w*eta)*kappa + 3 * h2*eta2*kappa*kappa);
		d_ws.df2[h_index] += scaled_cK_b3*invH*invH;

	}


}

#endif