// ----------------------------------------------------------------------------------
// Copyright 2016-2017 Michael-Angelo Yick-Hang Lam
//
// The development of this software was supported by  
// National Science Foundation Grant Number DMS-1211713.
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


#ifndef STATUS_LOGGER
#define STATUS_LOGGER

#include "parameters.h"
#include "output.h"

template <typename DATATYPE> class status_logger
{

public:


	status_logger()
	{
		time_step_count=0;
		n_status_count=0;
	}
	

	void add_entry( size_t t_index , DATATYPE dt , newton_status::status n_status , size_t n_count )
	{
		if( ( n_status == newton_status::SUCCESS ) || ( n_status == newton_status::INCREASE_DT ) )
		{


			auto it = this->n_count.end();
			this->n_count.insert(it , n_count);
			auto it2 = this->dt.end();
			this->dt.insert(it2 , dt);
			time_step_count++;
		}

		if(  n_status != newton_status::SUCCESS )
		{
			auto it = this->n_status.end();
			this->n_status.insert(it , n_status);
			auto it2 = this->t_index.end();
			this->t_index.insert(it2 , t_index);
			n_status_count++;
		}

	}

	void commit_data_to_files()
	{
		
		output_append_vector<size_t>( file_directories::newtonCountData , &n_count[0] , time_step_count );
		output_append_vector<DATATYPE>( file_directories::timestepData , &dt[0] , time_step_count  );
		
		output_append_vector<size_t>( file_directories::gaditStatusIndices , &t_index[0] , n_status_count );
		output_append_vector<newton_status::status >( file_directories::gaditStatus , &n_status[0] , n_status_count );

		this->clean();

	}




	
private:
	

	size_t time_step_count;
	vector<size_t> n_count;
	vector<DATATYPE> dt;

	
	size_t n_status_count;
	vector<newton_status::status> n_status;
	vector<size_t> t_index;
	
	void clean()
	{
		time_step_count=0;
		n_status_count=0;

		n_count.clear();
		dt.clear();
		n_status.clear();
		t_index.clear();
	}

};

#endif