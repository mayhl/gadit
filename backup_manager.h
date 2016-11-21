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
#include <chrono>
#include "parameters.h"


template <typename DATATYPE> class backup_manager
{
public:


	void initialize( backup_parameters parameters)
	{
		this->parameters = parameters;
		timeLastupdate =  std::chrono::system_clock::now();
	}

	bool is_backup_time()
	{
		currentTime  =  std::chrono::system_clock::now();

		long long timeDiff =  std::chrono::duration_cast< std::chrono::seconds >(currentTime - timeLastupdate).count();

		if ( timeDiff > parameters.updateTime )
		{
			timeLastupdate = currentTime;
			return true;
		}
		else
			return false;
	}

private:
	
	backup_parameters parameters;

	std::chrono::system_clock::time_point timeLastupdate;
	std::chrono::system_clock::time_point currentTime;
	

};
#endif