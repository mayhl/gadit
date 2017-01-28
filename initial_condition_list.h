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
// Name:			initial_condition.h
// Version: 		1.0
// Purpose:		Allows easy switching between different types of initial
//						conditions
// ----------------------------------------------------------------------------------

#ifndef INITIAL_CONDITION_LIST
#define INITIAL_CONDITION_LIST

#include "work_space.h"
#include "memory_manager.h"
#include "parameters.h"
#include "solver_template.h"
#include "output.h"

#define PI 3.14159265359


namespace ic_linear_waves
{
	initial_condition::id const ID = initial_condition::LINEAR_WAVES;
	template <typename DATATYPE > struct initial_parameters<DATATYPE, ID>
	{
		int nx;
		int ny;
		DATATYPE epy;
		DATATYPE epx;
		DATATYPE h0;

		std::string to_string()
		{

			std::string output;
			output = format_parameter_output::make_title("Linear Wave Initial Condition");

			output += "h0  = " + format_parameter_output::datatype(this->h0) + "\n";
			output += "nx  = " + format_parameter_output::datatype(this->nx) + "\n";
			output += "ny  = " + format_parameter_output::datatype(this->ny) + "\n";
			output += "epx = " + format_parameter_output::datatype(this->epx) + "\n";
			output += "epy = " + format_parameter_output::datatype(this->epy) + "\n";
			output += "\n";

			return output;
		}
	};

	template<typename DATATYPE, initial_condition::id IC_ID> DATATYPE 
		compute(DATATYPE &x, DATATYPE &y,
				spatial_parameters<DATATYPE> parasSpatial, initial_parameters<DATATYPE,IC_ID> parasInitial){ return 0;}

	template<typename DATATYPE, initial_condition::id IC_ID = ID> DATATYPE 
		compute(DATATYPE &x, DATATYPE &y, 
				spatial_parameters<DATATYPE> parasSpatial, initial_parameters<DATATYPE,ID> parasInitial)
	{
		return parasInitial.h0*( 1 + parasInitial.epx*( cos (parasInitial.nx*PI*x/(parasSpatial.xn-parasSpatial.x0) ) )
			                       + parasInitial.epy*( cos (parasInitial.ny*PI*y/(parasSpatial.ym-parasSpatial.y0) ) ) );	
	}



}

namespace initial_condition_list{

	enum load_status
	{
		SUCCESS,
		CAN_NOT_OPEN_FILE,
		NON_IMPLEMENTED_ID,
	};

	template<typename DATATYPE, initial_condition::id IC_ID> load_status compute(DATATYPE *x, DATATYPE *y, DATATYPE *h, dimensions dims , 
		spatial_parameters<DATATYPE> parasSpatial, initial_parameters<DATATYPE,IC_ID> parasInitial , io_parameters parasIo )
	{	


		if ( IC_ID == initial_condition::LOAD_FROM_FILE )
		{
			bool is_file_loaded;
			is_file_loaded = load_binary (parasIo.root_directory + file_directories::icInputFile , h , dims.n_pad , dims.m_pad );

			if ( is_file_loaded ) 
				return load_status::SUCCESS;
			else
				return load_status::CAN_NOT_OPEN_FILE;
		}
		else
		{

			bool is_invalid_id = false;
			for (int j = 0; j < dims.m ; j++)
			{
				for (int i = 0; i < dims.n; i++)
				{
					int k = (j +  dims.padding)*dims.n_pad +  dims.padding + i;

					// Add new initial conditions here.
					switch (IC_ID)
					{
					case initial_condition::LINEAR_WAVES:
						h[k] = ic_linear_waves::compute<DATATYPE, IC_ID>( x[i] , y[j] , parasSpatial , parasInitial);
						break;
					}
				}

				if ( is_invalid_id ) break;
			}

			if ( is_invalid_id )  
				return load_status::NON_IMPLEMENTED_ID;
			else
				return load_status::SUCCESS;

		}

	};

	// wrapper for memory_units to pass host pointers  
	template<typename DATATYPE, initial_condition::id IC_ID> load_status compute
		(memory_unit<DATATYPE> *x, memory_unit<DATATYPE> *y, memory_unit<DATATYPE> *h, dimensions dims, 
		spatial_parameters<DATATYPE> parasSpatial ,   initial_parameters<DATATYPE,IC_ID> parasInitial, io_parameters parasIo)
	{
		return compute<DATATYPE,IC_ID>(x->data_host, y->data_host, h->data_host, dims, parasSpatial, parasInitial, parasIo);
	};
}




#endif