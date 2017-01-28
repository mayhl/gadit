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

#ifndef MODEL_POLYMER
#define MODEL_POLYMER

// Imports base model_parameters template.
#include "solver_template.h"

namespace model_polymer
{
	// If copying to make new model, change to new model id.
	const model::id ID = model::POLYMER;

	template <typename DATATYPE > struct model_parameters<DATATYPE, ID>
	{

		// Main parameters
		DATATYPE cC;		

		DATATYPE  ci;
		DATATYPE  ASIO;
		DATATYPE  ASI;
		DATATYPE  d;

		// Derived parameters common to each spacial point,
		// precomputed on CPU to remove redundant GPU computations.
		// Only required if expression for nonlinear functions 
		// are complicated and contain many fixed terms.
	
		std::string to_string()
		{
			std::string output;
			output = format_parameter_output::make_title("Polymer Model");

			output += "cC   = " + format_parameter_output::datatype(this->cC) + "\n";
			output += "ci   = " + format_parameter_output::datatype(this->ci) + "\n";
			output += "ASIO = " + format_parameter_output::datatype(this->ASIO) + "\n";
			output += "ASI  = " + format_parameter_output::datatype(this->ASI) + "\n";
			output += "d    = " + format_parameter_output::datatype(this->d) + "\n";
			output += "\n";

			return output;
		}

		void compute_derived_parameters()
		{
		}

		DATATYPE getGrowthRate(DATATYPE h, DATATYPE q)
		{
			DATATYPE gRate;
			DATATYPE f1;
			DATATYPE f2;

			DATATYPE h3 = h*h*h;
			DATATYPE inv_h = 1.0 / h;

			DATATYPE inv_hd = 1.0 / (h + d);

			DATATYPE term1 = -72 * ci*pow(inv_h, 10);
			DATATYPE term2 = -6 * ASIO*pow(inv_h, 4);
			DATATYPE term3 = -6 * (ASIO - ASI)*pow(inv_hd, 4);

			DATATYPE term4 = term1 + term2 + term3;


			f2 = h3*term4;

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

			DATATYPE h3 = h*h*h;
			DATATYPE inv_h = 1.0 / h;

			DATATYPE inv_hd = 1.0 / (h + d);

			DATATYPE term1 = -72 * ci*pow(inv_h, 10);
			DATATYPE term2 = -6 * ASIO*pow(inv_h, 4);
			DATATYPE term3 = -6 * (ASIO - ASI)*pow(inv_hd, 4);

			DATATYPE term4 = term1 + term2 + term3;


			f2 = h3*term4;

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
		(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h , model_parameters<DATATYPE, MODEL_ID> modelParas, int h_index);
	

	// Definition of nonlinear functions for this model ID.
	template <typename DATATYPE, model::id MODEL_ID=ID> __device__ __forceinline__
		void nonlinear_functions(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h , model_parameters<DATATYPE, ID > modelParas, int h_index)
	{

		DATATYPE cC = modelParas.cC;

		DATATYPE h = d_h[h_index];
		DATATYPE h2 = h*h;
		DATATYPE h3 = h2*h;

		d_ws.f1[h_index] = cC*h3;

		d_ws.df1[h_index] = 3.0*cC*h2;

		DATATYPE  ci = modelParas.ci;
		DATATYPE  ASIO = modelParas.ASIO;
		DATATYPE  ASI = modelParas.ASI;
		DATATYPE  d = modelParas.d;

		DATATYPE inv_h = 1.0 / h;

		DATATYPE inv_hd = 1.0 / (h+d);
		
		DATATYPE term1 = -72*ci*pow(inv_h,10);
		DATATYPE term2 = - 6*ASIO*pow(inv_h,4 ) ;
		DATATYPE term3 =   -6*(ASIO-ASI)*pow(inv_hd,4);

		DATATYPE term4 = term1 + term2 + term3;


		d_ws.f2[h_index] = h3*term4;
		d_ws.df2[h_index] = 3.0*h2*term4 + h3*( term1*10*inv_h + term2*4*inv_h + term3*4*inv_hd );

	}


}

#endif