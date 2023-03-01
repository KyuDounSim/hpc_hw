#include <iostream>
#include <math.h>
#include <iterator>
#include <algorithm>
#include <vector>

#include "utils.h"

int main(int argc, char** argv) {

	if(argc < 2) {
		printf("At least one argument must be provided.\n");
		return -1;
	}
	Timer t;
	t.tic();

	int user_input = atoi(argv[1]);
	int max_iteration = -1;
	double residue_factor_limit = 1e-4;

	// two arguments provided
	if (argc == 3) {
		max_iteration = atoi(argv[2]);
	}

	double residue = 0.0, temp_sum = 0.0, lower_term = 0.0, higher_term = 0.0, initial_residue = 0.0,
		h = 1.0 / (user_input + 1), h_sqr = pow(h, 2);

	printf("User input is %d \n", user_input);

	double u_vec[user_input], u_vec_snap[user_input], f_vec[user_input];

	// initialize u_vec
	for(int i = 0; i < user_input; ++i) u_vec[i] = 0.0;

	// initialize u_vec_snap
	for(int i = 0; i < user_input; ++i) u_vec_snap[i] = 0.0;
	
	// initialize f_vec
	for(int i = 0; i < user_input; ++i) f_vec[i] = 1.0;

	// initial residue calculation
	for(int i = 0; i < user_input; ++i) {
		lower_term = (i - 1) < 0 ? 0.0 : u_vec[i - 1];
		higher_term = (i + 1) > user_input ? 0.0 : u_vec[i + 1];
		initial_residue += pow(( (-1.0 / h_sqr) * (lower_term + higher_term) + (2.0 / h_sqr) * u_vec[i] - f_vec[i]), 2);
	}

	initial_residue = sqrt(initial_residue); 
	printf("Initial residue : %f \n", initial_residue);

	unsigned long long int itr = 0;

	do {
		residue = 0.0;

		for(int i = 0; i < user_input; ++i) u_vec_snap[i] = u_vec[i];

		// residue calculation
		for(int i = 0; i < user_input; ++i) {
			lower_term = (i - 1) < 0 ? 0.0 : u_vec_snap[i - 1];
			higher_term = (i + 1) > user_input ? 0.0 : u_vec_snap[i + 1];
			residue += pow( (-1.0 / h_sqr) * (lower_term + higher_term) + (2.0 / h_sqr) * u_vec_snap[i] - f_vec[i], 2);
		}
	
		residue = sqrt(residue);

		printf("Iteration %llu's Norm Residue : %f \n", itr, residue);
		
		for(int i = 0; i < user_input; ++i) {
			lower_term = (i - 1) < 0 ? 0.0 : u_vec_snap[i - 1];
			higher_term = (i + 1) > user_input ? 0.0 : u_vec_snap[i + 1];

			temp_sum = (-1.0 / h_sqr) * (lower_term + higher_term);
			u_vec[i] = (h_sqr / 2.0 ) * (f_vec[i] - temp_sum);
		}

		++itr;
	
	} while((residue / initial_residue) > residue_factor_limit);

	double time = t.toc();

	printf("Final residue: %f\n", residue);
	printf("Total iteration: %llu\n", itr);
	printf("Time is %f \n", time);

	return 0;
}
