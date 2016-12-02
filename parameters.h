
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

namespace format_parameter_output
{
	std::string make_title(std::string title)
	{

		std::string output;
		std::string border;

		size_t length = title.length();

		border = std::string(length + 4, '-');


		output = border + "\n";
		output += "- " + title + " -\n";
		output += border + "\n\n";

		return output;

	}

	std::string datatype(int val)
	{
		std::string output;
		char buff[100];
		sprintf(buff, "%d" , val);
		output = buff;
		return output;
	}

	std::string datatype(size_t val)
	{
		std::string output;
		char buff[100];
		sprintf(buff, "%llu", val);
		output = buff;
		return output;
	}
	 std::string  datatype(double val)
	{

		std::string output;
		char buff[100];
		if (abs(val) < pow(10, -6))
			sprintf(buff, "%e", val);
		else if(abs(val) > pow(10, 6))
			sprintf(buff, "%e", val);
		else
			sprintf(buff, "%f", val);
		
		output = buff;
		return output;
	}

	 std::string  datatype(float val)
	{

		std::string output;
		char buff[100];
		if (abs(val) < pow(10, -6))
			sprintf(buff, "%e", val);
		else if (abs(val) > pow(10, 6))
			sprintf(buff, "%e", val);
		else
			sprintf(buff, "%f", val);

		output = buff;
		return output;
	}

}

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

	void compute_derived_parameters()
	{
		this->xn = ds*n;
		this->ym = ds*m;

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

	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Spatial");

		output += "ds = " + format_parameter_output::datatype(this->ds) + "\n";
		output += "n  = " + format_parameter_output::datatype(this->n) + "\n";
		output += "x0 = " + format_parameter_output::datatype(this->x0) + "\n";
		output += "xn = " + format_parameter_output::datatype(this->xn) + "\n";
		output += "m  = " + format_parameter_output::datatype(this->m) + "\n";
		output += "y0 = " + format_parameter_output::datatype(this->y0) + "\n";
		output += "ym = " + format_parameter_output::datatype(this->ym) + "\n";

		return output;
	}

};



template <typename DATATYPE> struct newton_parameters{
	DATATYPE error_tolerence;
	DATATYPE small_value;
	DATATYPE truncation_tolerance;
	
	int max_iterations;
	int min_iterations;	
	
	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Newton Iterations");

		output += "error_tolerence = " + format_parameter_output::datatype(this->error_tolerence) + "\n";
		output += "max_iterations  = " + format_parameter_output::datatype(this->max_iterations) + "\n";
		output += "min_iterations  = " + format_parameter_output::datatype(this->min_iterations) + "\n";

		return output;
	}

};

struct io_parameters
{
	std::string root_directory;
	bool is_full_text_output;
	bool is_console_output;
};

template <typename DATATYPE> struct temporal_parameters{
	
	DATATYPE t_start;
	DATATYPE t_end;

	DATATYPE dt_out;
	
	DATATYPE dt_min;
	DATATYPE dt_max;

	DATATYPE dt_init;
	
	DATATYPE dt_ratio_increase;
	DATATYPE dt_ratio_decrease;

	int min_stable_step;

	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Newton Iterations");

		output += "t_start           = " + format_parameter_output::datatype(this->error_tolerence) + "\n";
		output += "t_end             = " + format_parameter_output::datatype(this->t_end) + "\n";
		output += "dt_out            = " + format_parameter_output::datatype(this->dt_out) + "\n";
		output += "dt_min            = " + format_parameter_output::datatype(this->dt_min) + "\n";
		output += "dt_max            = " + format_parameter_output::datatype(this->dt_max) + "\n";
		output += "dt_init           = " + format_parameter_output::datatype(this->dt_init) + "\n";
		output += "min_stable_step   = " + format_parameter_output::datatype(this->min_stable_step) + "\n";
		output += "dt_ratio_increase = " + format_parameter_output::datatype(this->dt_ratio_increase) + "\n";
		output += "dt_ratio_decrease = " + format_parameter_output::datatype(this->dt_ratio_decrease) + "\n";

		return output;
	}

};
 struct backup_parameters{
	long long updateTime ;
	std::string to_string()
	{
		std::string output;
		output = format_parameter_output::make_title("Newton Iterations");

		output += "t_start           = " + format_parameter_output::datatype(this->error_tolerence) + "\n";
		output += "t_end             = " + format_parameter_output::datatype(this->t_end) + "\n";
		output += "dt_out            = " + format_parameter_output::datatype(this->dt_out) + "\n";
		output += "dt_min            = " + format_parameter_output::datatype(this->dt_min) + "\n";
		output += "dt_max            = " + format_parameter_output::datatype(this->dt_max) + "\n";
		output += "dt_init           = " + format_parameter_output::datatype(this->dt_init) + "\n";
		output += "min_stable_step   = " + format_parameter_output::datatype(this->min_stable_step) + "\n";
		output += "dt_ratio_increase = " + format_parameter_output::datatype(this->dt_ratio_increase) + "\n";
		output += "dt_ratio_decrease = " + format_parameter_output::datatype(this->dt_ratio_decrease) + "\n";

		return output;
	}
	
};

#endif