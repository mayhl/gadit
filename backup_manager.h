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

// ----------------------------------------------------------------------------------
// Name:	backup_manager.h
// Version: 1.0
// Purpose:	Encapsulate backing up of solution and relevant data.
// ----------------------------------------------------------------------------------

#ifndef BACKUP_MANAGER
#define BACKUP_MANAGER
#include <ctime>
#include "parameters.h"


template <typename DATATYPE> class backup_manager
{
public:


	void initialize(long long backup_time )
	{
		this->backup_time = backup_time;
		timeLastupdate = clock();
	}


	double get_time_elasped()
	{
		currentTime = clock();
		return (currentTime - timeLastupdate) / CLOCKS_PER_SEC/ 60;
	}

	double get_time_left()
	{	
		return backup_time - get_time_elasped();
	}

	bool is_backup_time()
	{
		currentTime  = clock();

		//long long timeDiff =  std::chrono::duration_cast< std::chrono::minutes >(currentTime - timeLastupdate).count();

		double timeDiff = get_time_elasped();

		if ( timeDiff > backup_time)
		{
			timeLastupdate = currentTime;
			return true;
		}
		else
			return false;
	}

private:
	
	long long backup_time;

	clock_t timeLastupdate;
	clock_t currentTime;
	

};
#endif