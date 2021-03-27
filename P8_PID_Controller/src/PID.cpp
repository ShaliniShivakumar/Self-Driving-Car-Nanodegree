#include "PID.h"
#include <limits>

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;
  
  p_error = 0.0;
  d_error = 0.0;
  i_error = 0.0;
  prev_cte = 0.0;
  counter = 0;
  error_sum = 0.0;
  
  max_error = std::numeric_limits<double>::max();
  min_error = std::numeric_limits<double>::min();	
}

void PID::UpdateError(double cte)
{
  /**
   * TODO: Update PID errors based on cte.
   */
  //prpportional error 
  p_error = cte;

  //diffrential error
  d_error = cte - prev_cte;
  prev_cte = cte;
	
  //integral error
  error_sum+= cte;
  i_error = error_sum;
  
  counter++;
  
  if (cte > max_error)
  {
    max_error = cte;
  }
  
  if(cte < min_error)
  {
    min_error = cte;
  }
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return Kp*p_error + Kd*d_error+ Ki*i_error;  // TODO: Add your total error calc here!
}

double PID::MaxError()
{
  return max_error;
}

double PID::MinError()
{
  return min_error;
}

double PID::AverageError()
{
  return error_sum/counter;
}