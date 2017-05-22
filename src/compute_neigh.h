/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(neigh,ComputeNeigh)

#else

#ifndef LMP_COMPUTE_NEIGH_H
#define LMP_COMPUTE_NEIGH_H

#include "compute.h"

#include <fstream>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class ComputeNeigh : public Compute {
 public:
  ComputeNeigh(class LAMMPS *, int, char **);
  void init();
  double compute_scalar();
  void init_list(int, class NeighList *);

 private:
  int myStep = 0;
  std::vector<int> chosenAtoms;
  std::string filename0;   // 0: Si, 1: O2
  std::string filename1;
  std::vector<double> alpha;
  std::vector<int> tau;
  int nTypes;

  std::ofstream outfiles[2];

  class NeighList *list;         // standard neighbor list used by most pairs
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
