
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

#ifndef PARAMETERS
#define PARAMETERS


#include <chrono>

namespace newton_status{
	enum status
	{
		SUCCESS = 0,
		CONVERGENCE_FAILURE_SMALL = 1,
		CONVERGENCE_FAILURE_LARGE = 2,
		NEGATIVE_SOLUTION = 3,
		TRUNCATION_ERROR = 4,
		INCREASE_DT = 255,
	};
}

namespace newton_stage
{
	enum stage
	{
		INITIAL,
		PRELOOP,
		LOOP,
	};
}

enum bc_postition
{
	FIRST, SECOND, THIRD, FIRST_LAST, SECOND_LAST, THIRD_LAST, INTERIOR,
};


enum boundary_condtion_type{
	SYMMETRIC,
	CONSTANT_FLOW,
};

template <typename DATATYPE> struct spatial_parameters{

	DATATYPE ds;

	DATATYPE inv_ds;
	DATATYPE inv_ds2;
	DATATYPE inv_ds4;

	DATATYPE scaled_inv_ds2;
	DATATYPE scaled_inv_ds4;
	
	size_t n;
	size_t m;

	DATATYPE x0;
	DATATYPE xn;
	DATATYPE y0;
	DATATYPE ym;

	void set_partition(DATATYPE ds, DATATYPE x0, DATATYPE xn, DATATYPE y0, DATATYPE ym)
	{
		this->ds = ds;
		this->x0 = x0;
		this->xn = xn;
		this->y0 = y0;
		this->ym = ym;

		inv_ds = 1.0 / ds;
		inv_ds2 = inv_ds*inv_ds;
		inv_ds4 = inv_ds2*inv_ds2;

		// absorbing all constants into derivative factors

		// 1) 1/2 factor from evaluating functions at have points i.e h(x-dx/2) = ( h(x)+h(x-dx) )/2 + O(dx^2)
		// 2) another 1/2 factor from Crank-Nicholson i.e theta and 1-theta for theta=1/2
		DATATYPE const prefactor = 0.5*0.5;

		scaled_inv_ds2 = prefactor*inv_ds2;
		scaled_inv_ds4 = prefactor*inv_ds4;
	}

};



template <typename DATATYPE> struct newton_parameters{
	DATATYPE error_tolerence;
	DATATYPE small_value;
	DATATYPE truncation_tolerance;
	
	int max_iterations;
	int min_iterations;

};



template <typename DATATYPE> struct temporal_parameters{
	
	DATATYPE t_start;
	DATATYPE t_end;

	DATATYPE dt_out;
	
	DATATYPE dt_min;
	DATATYPE dt_max;
	
	DATATYPE dt_ratio_increase;
	DATATYPE dt_ratio_decrease;

	DATATYPE dt_init;

	int min_stable_step;

};
 struct backup_parameters{
	long long updateTime ;
};

#endif