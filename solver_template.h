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

#ifndef SOLVER_TEMPLATE
#define SOLVER_TEMPLATE
#include "parameters.h"

namespace model
{
	enum id{
		DEFAULT,
		NLC,
		POLYMER,
		CONSANT,
	};
}
namespace initial_condition{

	enum id{
		LOAD_FROM_FILE,
		LINEAR_WAVES,
	};

}
// dummy definitions
template <typename DATATYPE, model::id MODEL_ID> struct model_parameters
{
	std::string to_string() { return ""; };
	void compute_derived_parameters() {};
};
template <typename DATATYPE, initial_condition::id IC_ID> struct initial_parameters
{
	std::string to_string() { return ""; };
};

template <typename DATATYPE, model::id MODEL_ID, initial_condition::id IC_ID> struct parameters
{
	model_parameters<DATATYPE, MODEL_ID>	model;
	initial_parameters<DATATYPE, IC_ID>	initial;
	spatial_parameters<DATATYPE>	spatial;
	temporal_parameters<DATATYPE>	temporal;
	newton_parameters<DATATYPE>		newton;
	backup_parameters				backup;
	io_parameters					io;

	std::string to_string()
	{
		std::string output;
		output = model.to_string();

		return output;
	}

};


#endif