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

#ifndef MODEL_CONSTANT
#define MODEL_CONSTANT

// Imports base model_parameters template.
#include "solver_template.h"

namespace model_constant
{
	// If copying to make new model, change to new model id.
	const model::id ID = model::CONSANT;

	template <typename DATATYPE > struct model_parameters<DATATYPE, ID>
	{

		// Main parameters
		DATATYPE c1;
		DATATYPE c2;

		// Derived parameters common to each spacial point,
		// precomputed on CPU to remove redundant GPU computations.
		// Only required if expression for nonlinear functions 
		// are complicated and contain many fixed terms.

		std::string to_string()
		{
			std::string output;
			output = format_parameter_output::make_title("NLC Model");

			output += "c1   = " + format_parameter_output::datatype(this->c1) + "\n";
			output += "c2   = " + format_parameter_output::datatype(this->c2) + "\n";
			output += "\n";

			return output;
		}

		void compute_derived_parameters()
		{
		}

		DATATYPE getGrowthRate(DATATYPE h, DATATYPE q)
		{
			DATATYPE gRate;

			DATATYPE q2 = q*q;
			gRate = -(c1*q2 - c2)*q2;

			return gRate;


		}

		DATATYPE getMaxGrowthMode(DATATYPE h)
		{
			DATATYPE qm;

			qm = sqrt(c2 / (2.0*c1));

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

		d_ws.f1[h_index] = modelParas.c1;
		d_ws.df1[h_index] = 0.0;


		d_ws.f2[h_index] = modelParas.c2;
		d_ws.df2[h_index] = 0.0;

	}


}

#endif