// ----------------------------------------------------------------------------------
// Copyright 2016-2017 Michael-Angelo Yick-Hang Lam
//
// The development of GADIT was supported by the National Science Foundation
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

#ifndef TIMESTEP_MANAGER
#define TIMESTEP_MANAGER


namespace timestep_manager_status
{
	enum status
	{
		SUCCESS,
		MIN_DT,
		DT_CHANGE_OUTPUT,
	};
}

template <typename DATATYPE> class timestep_manager
{

public:

	void initialize( temporal_parameters<DATATYPE> parameters )
	{
		this->parameters		= parameters;

		dt						= parameters.dt_init;
		dt_temp					= parameters.dt_init;
		t						= parameters.t_start;
		t_out					= t + parameters.dt_out;
		t_next					= t + dt;

		isOutputStep			= false;
		isSuccessfulOutputStep	= false;
		isNotCompleted			= true;
		is_at_max_time_step		= false;
		outputStep				= 0;
		stable_time_steps		= 0;
		time_steps				= 0;
	}


	bool is_not_completed()
	{
		return isNotCompleted;
	}

	size_t get_iteration_index()
	{
		return time_steps;
	}

	size_t get_next_output_index()
	{
		return outputStep;
	}

	DATATYPE get_current_time()
	{
		return t;
	}


	// returns false only if dt goes below dt_min.
	timestep_manager_status::status update_dt(  newton_status::status &status )
	{
		
		isSuccessfulOutputStep = false;
		if ( status == newton_status::SUCCESS  )
		{
	
			time_steps++;
			stable_time_steps++;

			t = t_next;
			increase_dt(status);

		}
		else
		{
			stable_time_steps = 0;
			is_at_max_time_step = false;
			// Unexpected state, dt was lowered to match output time but failed.
			if ( isOutputStep ) return timestep_manager_status::DT_CHANGE_OUTPUT;

			dt /= parameters.dt_ratio_decrease;
			if ( dt < parameters.dt_min ) return timestep_manager_status::MIN_DT;
		}
		
		return timestep_manager_status::SUCCESS;
	}


	bool is_sucessful_output_step()
	{
		return isSuccessfulOutputStep;
	}

	DATATYPE get_timestep()
	{
		return dt;
	}


private:
	DATATYPE dt; 
	DATATYPE dt_temp;
	
	DATATYPE t;
	DATATYPE t_next;
	DATATYPE t_out;
	
	bool isOutputStep;
	bool isSuccessfulOutputStep;
	bool isNextOutputStep;

	bool isNotCompleted;

	bool is_at_max_time_step = false;

	size_t outputStep;
	size_t stable_time_steps;
	size_t time_steps;
	
	temporal_parameters<DATATYPE> parameters;
	
	void increase_dt( newton_status::status &status )
	{
		
		if ( !isOutputStep )
		{
				
			// Increasing dt within dt_max
			if( (stable_time_steps >= parameters.min_stable_step) )
			{
				dt*=parameters.dt_ratio_increase;
				status = newton_status::INCREASE_DT;

				if ( dt > parameters.dt_max )
				{
					dt = parameters.dt_max;
					is_at_max_time_step = true;
				}

			}


			// Adjusting dt temporarily if close to t_out
			t_next = t + dt;

			if ( t_next >= t_out )
			{
				dt_temp = dt;

				dt = t_out - t;
				t_next = t_out;

				isOutputStep = true;
			}

		}
		else
		{
			
			outputStep++;

			// switching dt back from temporary output dt
			dt = dt_temp;

			t_next = t_out + dt;

			t_out = (outputStep+1)*parameters.dt_out;

			isOutputStep = false;
			isSuccessfulOutputStep = true;

			if (t_out > parameters.t_end ) isNotCompleted = false;
	
		}
	}


};

#endif