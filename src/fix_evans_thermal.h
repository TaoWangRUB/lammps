/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(evans/thermal,FixEvansThermal)

#else

#ifndef LMP_FIX_EVANS_THERMAL_H
#define LMP_FIX_EVANS_THERMAL_H

#include "fix.h"

namespace LAMMPS_NS {

class FixEvansThermal : public Fix {
 public:
  FixEvansThermal(class LAMMPS *, int, char **);
  ~FixEvansThermal();
  /** modify parameters after construction */
  int modify_param(int narg, char** arg);
  /** tell LAMMPS which fix methods to call */
  int setmask();
  /** set-up before "run" */
  virtual void init();
  /** first Verlet step */
  virtual void initial_integrate(int);
  /** second Verlet step */
  virtual void final_integrate();
  /** output to thermo */
  virtual double compute_vector(int n);
  /** communicate change of timestep */
  void reset_dt();
    
  void init_list(int, class NeighList *);

 protected:
  /** computes for PE and virial */
  class Compute *cPE_,*cVirial_;
  /** timesteps */
  double dtv,dtf,dtq;
  /** flag for computing Evans' forces on subdomains */
  bool parallelPartition_;
  /** compute system variables, returns temperature */
  double compute_thermo(void); 
  /** thermostat */
  bool useThermostat_;
  /** thermostat multiplier */
  double lambda_;
  /** actual & set temperature */
  double T_, T0_;
  /** Nose thermostat 1/period */
  double noseFreq_;
  /** step */
  int step_;
  /** field strength = d ln T / d x  */
  double fieldX_, fieldY_, fieldZ_;
  /** average D.F */
  double aveDFx_, aveDFy_, aveDFz_;
  /** total mass */
  double M_,Mlocal_;
  /**  number of atoms in group */
  double N_,Nlocal_;
  /** heat flux */
  double J_[3];
  /** center of mass velocity */
  double vBar_[3], vBarLocal_[3];
    
    
  /** neighbor list pointer */
  class NeighList *list_;

};

}

#endif
#endif
