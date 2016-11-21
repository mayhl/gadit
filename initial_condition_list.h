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
// Name:			initial_condition_list.h
// Version: 		1.0
// Purpose:		Allows easy switching between different types of initial
//						conditions
// ----------------------------------------------------------------------------------

#ifndef INITIAL_CONDITION
#define INITIAL_CONDITION

#include "work_space.h"
#include "memory_manager.h"
#include "parameters.h"
#include "output.h"

namespace inital_condition_list{

	enum ic_id{
		LOAD_FROM_FILE,
		TEST,
	};

	template<typename DATATYPE> void compute_ic(DATATYPE *x, DATATYPE *y, DATATYPE *h, dimensions dims , spatial_parameters<DATATYPE> paras, ic_id  id)
	{	
		#define PI 3.14159265359

		switch (id)
		{
		case ic_id::LOAD_FROM_FILE:
			load_binary ( file_directories::icInputFile , h , dims.n_pad , dims.m_pad );
			break;

		case ic_id::TEST:
			for (int j = 0; j < dims.m ; j++)
			{
				for (int i = 0; i < dims.n; i++)
				{
					int k = (j +  dims.padding)*dims.n_pad +  dims.padding + i;
					h[k] = 0.2*( 1 - 0.1*( cos ( 80*PI*x[i]/(paras.xn-paras.x0) ) + ( cos ( 80*PI*y[j]/(paras.ym-paras.y0) ) ) ) );	
				}
			}
			break;

		default:
			break;
		}
	};

	// wrapper for memory_units to pass host pointers  
	template<typename DATATYPE> void compute_ic(memory_unit<DATATYPE> *x, memory_unit<DATATYPE> *y, memory_unit<DATATYPE> *h, dimensions dims, spatial_parameters<DATATYPE> paras, ic_id id)
	{
		compute_ic(x->data_host, y->data_host, h->data_host, dims, paras, id);
	};
}




#endif