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

#ifndef NONLINEAR_FUNCTIONS
#define NONLINEAR_FUNCTIONS

#include "cuda_runtime.h"
#include "work_space.h"
#include "parameters.h"
#include "boundary_conditions.h"

enum model_id{
	DEFAULT,
	NLC,

};

// dummy definition
template <typename DATATYPE, model_id id> struct model_parameters{};

template <typename DATATYPE, model_id id> struct parameters
{
	model_parameters<DATATYPE, id>	model;
	spatial_parameters<DATATYPE>	spatial;
	temporal_parameters<DATATYPE>	temporal;
	newton_parameters<DATATYPE>		newton;
	backup_parameters				backup;

};


#endif