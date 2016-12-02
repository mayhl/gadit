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

#ifndef MODEL_LIST
#define MODEL_LIST

#include "work_space.h"

// List of Models 
#include "model_default.h"
#include "model_nlc.h"
#include "model_polymer.h"
#include "model_constant.h"

namespace model_list
{
	template <typename DATATYPE,model::id MODEL_ID> __device__ __forceinline__
		void select_functions(reduced_device_workspace<DATATYPE> d_ws, DATATYPE *d_h , model_parameters<DATATYPE, MODEL_ID> modelParas, int index )
	{
		switch (MODEL_ID)
		{
		case model::DEFAULT:
			model_default::nonlinear_functions<DATATYPE>(d_ws, d_h , modelParas, index);
			break;
		case model::NLC:
			model_nlc::nonlinear_functions<DATATYPE>(d_ws, d_h, modelParas, index);
			break;
		case model::POLYMER:
			model_polymer::nonlinear_functions<DATATYPE>(d_ws, d_h, modelParas, index);
			break;
		case model::CONSANT:
			model_constant::nonlinear_functions<DATATYPE>(d_ws, d_h, modelParas, index);
			break;
		}
	}
};


#endif